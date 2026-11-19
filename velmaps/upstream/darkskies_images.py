from swiftsimio import load
from swiftsimio.visualisation.projection import project_gas
from swiftsimio.visualisation.projection import project_pixel_grid
from swiftsimio.visualisation.smoothing_length.generate import generate_smoothing_lengths

from project_pixel_dict import project_pixel_grid as project_pixel_grid_dict
from xray_interpolator import XrayCalculator
import h5py
import numpy as np
import unyt
from copy import deepcopy as cp
from contextlib import contextmanager
import pickle as pkl

from halo_cls import halo
from validate_halo import validate_halo, get_halo_idx, back_trace_halo
import os

'''
Units convention (changed from the previous version)
----------------------------------------------------
SWIFT snapshot coordinates are COMOVING (a-scale exponent 1), and the
VELOCIraptor catalogue is PHYSICAL (Comoving_or_Physical = 0).

- postage_stamp is now a PHYSICAL size. The region projected in the
  snapshot is postage_stamp / a in comoving Mpc, so every map is
  postage_stamp physical across and the pixel size is
  postage_stamp / resolution physical (20 pkpc for 2 pMpc / 100) at every z.
- Mass maps (total, dark matter, stellar) are returned as PHYSICAL surface
  densities in Msun / pMpc^2. The old maps were mass per (2 cMpc)^2, which is
  why add_shear_to_data.py divided darkskies norms by 4. That /4 must be
  REMOVED for maps made with this version.
- meta_data records the scale factor, pixel size, projection depth and the
  mean background surface density through the projection, so the lensing
  code can subtract the background consistently.
'''


def rebin(a, shape):
    '''
    Re bin an array in to a new shape, and take the average
    '''
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)


def to_mpc(value):
    '''Strip units, returning a float (or array) in Mpc. Bare numbers are assumed to be Mpc.'''
    if hasattr(value, 'units'):
        return np.asarray(value.to(unyt.Mpc).value)
    return np.asarray(value)


def permute_for_projection(coordinates, project_axis='z'):
    '''
    Return a COPY of the coordinates with the projection axis moved to the
    third column. Never modifies the snapshot in place, so repeated calls on
    the same snapshot give the same projection.
    '''
    coords = cp(coordinates)
    if project_axis == 'x':
        coords[:,0] = coordinates[:,1]
        coords[:,1] = coordinates[:,2]
        coords[:,2] = coordinates[:,0]
    elif project_axis == 'y':
        coords[:,1] = coordinates[:,2]
        coords[:,2] = coordinates[:,1]
    return coords


@contextmanager
def projected_coordinates(dataset, project_axis='z'):
    '''
    Temporarily permute a swiftsimio dataset's coordinates for projection and
    always restore them afterwards (avoids copying the whole dataset).
    '''
    original = dataset.coordinates
    cosmo_factor = getattr(original, 'cosmo_factor', None)
    permuted = permute_for_projection(original, project_axis)
    if cosmo_factor is not None:
        permuted.cosmo_factor = cosmo_factor
    dataset.coordinates = permuted
    try:
        yield dataset
    finally:
        dataset.coordinates = original


def get_range(center, comoving_stamp, projection_depth, project_axis='z'):
    '''
    Projection region in COMOVING Mpc, in the permuted frame (projection axis last).

    center : comoving Mpc
    comoving_stamp : comoving width of the map
    projection_depth : comoving depth along the line of sight
    '''
    center = to_mpc(center)
    half_w = to_mpc(comoving_stamp)/2.
    half_d = to_mpc(projection_depth)/2.

    order = {'x':(1, 2, 0), 'y':(0, 2, 1), 'z':(0, 1, 2)}[project_axis]
    c0, c1, c2 = center[list(order)]

    return np.array([c0-half_w, c0+half_w,
                     c1-half_w, c1+half_w,
                     c2-half_d, c2+half_d])*unyt.Mpc


def to_physical_surface_density(raw_map, comoving_stamp, scale_factor):
    '''
    project_pixel_grid normalises the map to the unit square of the region,
    so raw = Sigma_comoving * width^2 with width in comoving Mpc
    (the old /4 for a 2 cMpc stamp). Physical Sigma = Sigma_comoving / a^2.
    Returns Msun / pMpc^2 given masses already in Msun.
    '''
    width = to_mpc(comoving_stamp)
    return raw_map / width**2 / scale_factor**2


def get_halo_center(
        snapfile,
        props_file,
        halo_index,
        scale_factor,
        cdm_set,
        vr_center=True,
    ):
    '''
    Return the halo centre in COMOVING Mpc and the halo's catalogue properties.
    '''

    if vr_center:
        props = h5py.File(props_file, 'r')

        # VELOCIraptor positions are physical; snapshot coordinates are comoving
        xcenter = props['Xcmbp'][halo_index]/scale_factor
        ycenter = props['Ycmbp'][halo_index]/scale_factor
        zcenter = props['Zcmbp'][halo_index]/scale_factor

        center = np.array([xcenter, ycenter, zcenter])*unyt.Mpc
    else:
        props={}
        dirname=os.path.dirname(snapfile).replace('snap','halo')
        filename=snapfile.replace("snap","halo").replace("hdf5","pkl")
        if os.path.isfile(filename):
            this_halo = pkl.load(open(filename,'rb'))
        else:
            this_halo = halo(snapfile,nIterations=10)
            this_halo.get_all_density_profiles()
            del this_halo.coords
            del this_halo.mp
            if not os.path.isdir(dirname):
                os.system(f"mkdir -p {dirname}")
            # Cache every halo (previously only saved when the directory was new)
            pkl.dump(this_halo,open(filename,'wb'))

        # NB: check halo_cls returns this in comoving Mpc (snapshot units)
        center = this_halo.center_3d
        if not hasattr(center, 'units'):
            center = np.asarray(center)*unyt.Mpc

    return_props = {}
    for ikey in props.keys():
        try:
            return_props[ikey] = props[ikey][halo_index]
        except:
            continue
    if vr_center:
        props.close()
    return center, return_props


def get_images(
        snapfile,
        props_file,
        groups_file,
        cdm_set,
        postage_stamp=2*unyt.Mpc,
        resolution=100,
        project_axis='z',
        backend="histogram",
        total=True,
        dm=True,
        xray=True,
        stellar=True,
        bcg_shape=False,
        vr_center=True,
        projection_depth=400*unyt.Mpc,
    ):
    '''
    postage_stamp : PHYSICAL width of the map
    projection_depth : COMOVING depth of the projection
    Mass maps are returned in Msun / pMpc^2.
    '''

    stacked = []
    snap = load(snapfile)
    scale_factor = float(snap.metadata.a)

    comoving_stamp = postage_stamp/scale_factor
    pixel_size = postage_stamp/resolution
    print(f"Map will have a pixel size of {pixel_size.to(unyt.kpc)} (physical), "
          f"{(comoving_stamp/resolution).to(unyt.kpc)} (comoving)")

    if vr_center:
        intersections, halo_index = back_trace_halo(
        cdm_set, groups_file, props_file,
        nchecks=10,
        maxchecks=100,
        min_particles = 1000)
    else:
        halo_index = 0

    print("HALO INDEXES IS ",halo_index)
    center, props = get_halo_center(
        snapfile,
        props_file,
        halo_index,
        scale_factor,
        cdm_set,
        vr_center=vr_center
    )
    region = get_range(center, comoving_stamp, projection_depth, project_axis=project_axis)

    project_kwargs = dict(
        resolution=resolution,
        project_axis=project_axis,
        comoving_stamp=comoving_stamp,
        scale_factor=scale_factor,
    )

    if total:
        stacked.append(get_total_matter(snap, region, **project_kwargs))
    if dm:
        stacked.append(get_dark_matter(snap, region, backend=backend, **project_kwargs))
    if xray:
        stacked.append(get_xray_luminosity(snap, region, resolution=resolution,
                                           project_axis=project_axis))
    if stellar:
        stellar_map = get_stellar_matter(snap, region, backend=backend, **project_kwargs)
        stacked.append(stellar_map)

    # Mean matter surface density through the projection, physical Msun / pMpc^2:
    # rho_m(z) * physical depth = rho_m0 / a^3 * (depth_c * a) = rho_m0 * depth_c / a^2
    cosmology = snap.metadata.cosmology
    rho_m0 = (cosmology.critical_density0*cosmology.Om0).to('Msun/Mpc3').value

    meta_data = {}
    meta_data['redshift'] = int(snap.metadata.redshift*1e3)/1e3
    meta_data['scale_factor'] = scale_factor
    meta_data['props'] = props
    meta_data['map_units'] = 'Msun/pMpc^2'
    meta_data['pixel_size_pkpc'] = float(pixel_size.to(unyt.kpc).value)
    meta_data['postage_stamp_pMpc'] = float(to_mpc(postage_stamp))
    meta_data['projection_depth_cMpc'] = float(to_mpc(projection_depth))
    meta_data['background_sigma'] = rho_m0*float(to_mpc(projection_depth))/scale_factor**2

    if bcg_shape:
        if resolution < 1000:
            hi_res_kwargs = dict(project_kwargs, resolution=1000)
            hi_res_stellar_map = get_stellar_matter(snap, region, backend=backend, **hi_res_kwargs)
        else:
            hi_res_stellar_map = stellar_map
        # The map covers the full postage stamp, so pass that (physical) size
        meta_data['bcg_shape_e'] = get_BCG_shape(
            hi_res_stellar_map,
            postage_stamp=postage_stamp,
            radius=200*unyt.kpc)

    return np.array(stacked), meta_data


def get_stellar_matter( snap,
                       region,
                       resolution=100,
                       project_axis='z',
                       comoving_stamp=2*unyt.Mpc,
                       scale_factor=1.,
                       backend="histogram",
                      ):
    print("BACKEND IS %s" % backend)

    with projected_coordinates(snap.stars, project_axis) as stars:
        # Only the smoothing backends use smoothing lengths
        if backend != "histogram":
            stars.smoothing_length = generate_smoothing_lengths(
                        stars.coordinates,
                        snap.metadata.boxsize,
                        kernel_gamma=1.8,
                        neighbours=5,
                        speedup_fac=2,
                        dimension=3,
                    )

        stellar_mass = project_pixel_grid(
                    data=stars,
                    boxsize=snap.metadata.boxsize,
                    resolution=resolution,
                    project="masses",
                    region=region,
                    periodic=True,
                    backend=backend
                )

    return to_physical_surface_density(stellar_mass*1e10, comoving_stamp, scale_factor)


def get_dark_matter( snap,
                     region,
                     resolution=100,
                     project_axis='z',
                     comoving_stamp=2*unyt.Mpc,
                     scale_factor=1.,
                     backend="histogram",
                    ):
    print("BACKEND IS %s" % backend)

    with projected_coordinates(snap.dark_matter, project_axis) as dark_matter:
        if backend != "histogram":
            dark_matter.smoothing_length = generate_smoothing_lengths(
                        dark_matter.coordinates,
                        snap.metadata.boxsize,
                        kernel_gamma=1.8,
                        neighbours=5,
                        speedup_fac=2,
                        dimension=3,
                    )

        darkmatter_mass = project_pixel_grid(
                    data=dark_matter,
                    boxsize=snap.metadata.boxsize,
                    resolution=resolution,
                    project="masses",
                    region=region,
                    periodic=True,
                    backend=backend
                )

    return to_physical_surface_density(darkmatter_mass*1e10, comoving_stamp, scale_factor)


def get_total_matter( snap,
                     region,
                     resolution=100,
                     project_axis='z',
                     comoving_stamp=2*unyt.Mpc,
                     scale_factor=1.,
                    ):

    coordinates = np.vstack([
        snap.dark_matter.coordinates,
        snap.gas.coordinates,
        snap.stars.coordinates,
        snap.black_holes.coordinates,
        snap.boundary.coordinates
        ])

    masses = np.concatenate([
        snap.dark_matter.masses,
        snap.gas.masses,
        snap.stars.masses,
        snap.black_holes.dynamical_masses,
        snap.boundary.masses
        ])

    coordinates = permute_for_projection(coordinates, project_axis)

    smoothing_lengths = generate_smoothing_lengths(
                coordinates,
                snap.metadata.boxsize,
                kernel_gamma=1.6,
                neighbours=100,
                speedup_fac=2,
                dimension=3,
            )

    data = {'coordinates':coordinates, 'masses':masses,'smoothing_lengths':smoothing_lengths}
    total_mass= project_pixel_grid_dict(
        data,
        snap.metadata.boxsize,
        resolution=resolution,
        region=region,
        project="masses",
    )

    return to_physical_surface_density(total_mass*1e10, comoving_stamp, scale_factor)


def get_xray_luminosity( snap,
                        region,
                        resolution=100,
                        project_axis='z',
                        xray_redshift=0.,
                       ):
    '''
    xray_redshift : redshift used for the X-ray tables. 0 gives rest-frame
    luminosities (the previous behaviour); pass snap.metadata.redshift for
    the observed band.
    '''

    mass_fractions = np.array([
        snap.gas.element_mass_fractions.carbon,
        snap.gas.element_mass_fractions.helium,
        snap.gas.element_mass_fractions.hydrogen,
        snap.gas.element_mass_fractions.iron,
        snap.gas.element_mass_fractions.magnesium,
        snap.gas.element_mass_fractions.neon,
        snap.gas.element_mass_fractions.nitrogen,
        snap.gas.element_mass_fractions.oxygen,
        snap.gas.element_mass_fractions.silicon,]).T

    table_name = 'X_Ray_Tables.hdf5'
    redshifts = np.full(mass_fractions.shape[0], xray_redshift)
    band = ['ROSAT'] #Define which broadband to interpolate. Choices are ROSAT (0.5-2.0 keV), eRosita-low (0.2-2.3 keV), and eRosita-high (2.3-8.0 keV)
    observing_type = ['energies_intrinsic'] #choose to record either energy ['energies_intrinsic'] (erg/s) or photon count ['photons_intrinsic' (photons/sec)

    xray_calc = XrayCalculator(redshifts, table_name, band, observing_type)

    idx_z, idx_he, idx_T, idx_n, \
        t_z, d_z, t_T, d_T, t_nH, d_nH, t_He, d_He, \
        abundance_to_solar, joint_mask, volumes, data_n = \
            xray_calc.find_indices(
                snap.gas.densities,
                snap.gas.temperatures,
                mass_fractions,
                snap.gas.masses,
                redshifts,
                fill_value = 0
            )

    xray_luminosities = \
        xray_calc.interpolate_X_Ray(
            idx_z, idx_he, idx_T, idx_n,
            t_z, d_z, t_T, d_T, t_nH, d_nH, t_He, d_He,
            abundance_to_solar, joint_mask, volumes, data_n,
            bands = band,
            observing_types = observing_type,
            fill_value = 0)

    snap.gas.xray_luminosities = xray_luminosities.T[0,:]/1e38

    # project_gas needs the snapshot object, so permute the gas coordinates in place temporarily
    with projected_coordinates(snap.gas, project_axis):
        xray_lum_map = project_gas(
            snap,
            resolution=resolution,
            project='xray_luminosities',
            parallel=True,
            periodic=True,
            region=region,
        )

    return xray_lum_map


def get_BCG_shape( stellar_image,
                   postage_stamp=2*unyt.Mpc,
                   radius=200*unyt.kpc):
    '''
    postage_stamp must be the PHYSICAL width covered by stellar_image.
    '''

    pixels = stellar_image.shape[0]
    stamp_kpc = postage_stamp.to(unyt.kpc).value
    resolution = stamp_kpc/pixels

    x = np.arange(pixels)*resolution - stamp_kpc/2. + resolution/2.
    xg, yg = np.meshgrid(x, x)

    r = np.sqrt(xg**2 + yg**2)
    inside = r < radius.to(unyt.kpc).value

    bcg_shape_xx = np.sum( xg[inside]*xg[inside]*stellar_image[inside])
    bcg_shape_yy = np.sum( yg[inside]*yg[inside]*stellar_image[inside])
    bcg_shape_xy = np.sum( xg[inside]*yg[inside]*stellar_image[inside])
    bcg_shape_e1 = (bcg_shape_xx - bcg_shape_yy) / (bcg_shape_xx + bcg_shape_yy)
    bcg_shape_e2 = 2*bcg_shape_xy / (bcg_shape_xx + bcg_shape_yy)
    bcg_shape_e = np.sqrt( bcg_shape_e1**2+bcg_shape_e2**2)

    return bcg_shape_e


def append_meta_data( snapfile, props_file, groups_file, cdm_set, oldfile,
                      postage_stamp=2*unyt.Mpc, project_axis='z',
                      vr_center=True, projection_depth=400*unyt.Mpc):
    '''
    Add the BCG ellipticity to an existing image file without remaking the maps.
    '''

    snap = load(snapfile)
    scale_factor = float(snap.metadata.a)
    comoving_stamp = postage_stamp/scale_factor
    images = pkl.load(open(oldfile,'rb'))

    if vr_center:
        intersections, halo_index = back_trace_halo(
            cdm_set, groups_file, props_file,
            nchecks=10, maxchecks=100, min_particles=1000)
    else:
        halo_index = 0

    center, props = get_halo_center(
        snapfile, props_file, halo_index, scale_factor, cdm_set, vr_center=vr_center)
    region = get_range(center, comoving_stamp, projection_depth, project_axis=project_axis)

    hi_res_stellar_map = get_stellar_matter( snap,
                                     region,
                                     resolution=2000,
                                     project_axis=project_axis,
                                     comoving_stamp=comoving_stamp,
                                     scale_factor=scale_factor,
                                     )

    props['bcg_ell'] = get_BCG_shape( hi_res_stellar_map,
                     postage_stamp=postage_stamp,
                     radius=200*unyt.kpc)

    return images, props
