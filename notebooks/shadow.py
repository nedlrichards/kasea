import numpy as np
import numexpr as ne
from math import pi
import matplotlib.pyplot as plt
from src import XMitt, Realization, integral_mask

from scipy.interpolate import griddata

plt.ion()

toml_file = 'notebooks/sine.toml'
xmitt = XMitt(toml_file)

surf = xmitt.surface
realization = surf.gen_realization()

# work with mmap class
surface_realization = Realization(surf)
surface_realization.synthesize(0.)

distance_mask, proj_d_src, th_src, phi_src, u_i = integral_mask(surface_realization, xmitt.theta[0], xmitt.broadcast, to_shadow=True)
eta = surface_realization()[0].copy()


def fix_range(phi, rho, th, d_phi, rho_intrp):
    """quantize samples to a constant rho value and phi spacing"""
    p_0, p_1 = (phi[0], phi[-1])
    num_phi = int(np.ceil((phi[-1] - phi[0] - d_phi) / d_phi))
    # linear interpolator requires a sample above and below target
    rho_b = np.where(rho < rho_intrp)[0]
    rho_u = np.where(rho > rho_intrp)[0]
    p_0 = phi[max(rho_b[0], rho_u[0])]
    p_1 = phi[min(rho_b[-1], rho_u[-1])]

    phi_a = np.linspace(p_0, p_1, num_phi)[:, None]
    if phi_a.size < 1:
        1/0

    xi = np.concatenate([np.full_like(phi_a, rho_intrp), phi_a], axis=1)
    points = np.concatenate([rho[:, None], phi[:, None]], axis=1)
    #theta = griddata(points, th[:, None], xi, method='nearest')
    if th.size > 4:

        theta = griddata(points, th[:, None], xi, method='linear')

        # edges can give rise to nans
        if np.any(np.isnan(theta)):
            1/0

    if th.size <= 4 or theta.size == 0:
        phi_a = phi_a[:, 0]
        theta = griddata(points, th[:, None], xi, method='nearest')
        theta = theta[:, 0]
    return phi_a[:, 0], theta[:, 0]

# shadow check could have a larger scale than dx
s_dx = 10 * surf.dx

# section distance from source into bins 2 * s_dx wide
i_0 = np.argmax(proj_d_src > s_dx)
d_0 = proj_d_src[i_0]
i_1 = np.argmax(proj_d_src[i_0:] > d_0 + 2 * surf.dx) + i_0

i = 0
min_theta = np.array([[-np.pi, np.pi], [1, 1]])

fig, ax = plt.subplots()
cnt = i_0
shadow_record = [np.zeros(i_0, dtype=np.bool_)]

while i_0 < i_1:
    rho_range = proj_d_src[i_0: i_1]
    phi_range = phi_src[i_0: i_1]
    th_range = th_src[i_0: i_1]

    # look for gaps in phi
    phi_si = np.argsort(phi_range, kind='heapsort')
    phi_srt = phi_range[phi_si]
    rho_srt = rho_range[phi_si]
    th_srt = th_range[phi_si]


    # intepolate theta, phi onto constant rho contour
    rho_intrp = d_0 + surf.dx
    d_phi =  s_dx / rho_intrp
    gap_i = np.concatenate([[False], np.diff(phi_srt) > d_phi])
    gaps = np.where(gap_i)[0]

    if gap_i[-1] == False:
        gaps = np.concatenate([[0], gaps, [phi_srt.size]])
    else:
        # special case where gap hits edge
        gaps = np.concatenate([[0], gaps])

    # determine shadowing for each segment
    new_min_theta = []
    shadow_inds = []
    for g0, g1 in zip(gaps[:-1], gaps[1:]):
        phi_a, theta = fix_range(phi_srt[g0:g1], rho_srt[g0:g1], th_srt[g0:g1], d_phi, rho_intrp)
        if len(phi_a.shape) == 2:
            1/0
        pnts = np.array([phi_a, theta])

        # interpolate last theta front onto segment
        last_theta = np.interp(pnts[0], min_theta[0], min_theta[1])

        # shadow check with downsampled postion
        shadow_i = pnts[1] > last_theta

        # upsample shadows to full position record
        lookup = np.digitize(phi_srt[g0:g1], pnts[0])
        lookup[lookup == pnts.shape[1]] -= 1
        #shadow_inds = shadow_i[lookup]

        shadow_inds.append((phi_srt[g0:g1] > -0.3) & (phi_srt[g0:g1] < 0.3))

        # update theta minima estimate
        seg_min_theta = pnts.copy()
        seg_min_theta[1, shadow_i] = last_theta[shadow_i]

        new_min_theta.append(seg_min_theta)

    shadow_inds = np.concatenate(shadow_inds)

    # make a record of which positions are shadowed
    shadow_record.append(shadow_inds[np.argsort(phi_si)])

    # keep record of theta from between segments
    inter_theta = []

    inter_theta_i = min_theta[0] < new_min_theta[0][0, 0]
    inter_theta.append(min_theta[:, inter_theta_i])

    if len(new_min_theta) > 1:
        for mt0, mt1 in zip(new_min_theta[:-1], new_min_theta[1:]):
            inter_theta_i = (min_theta[0] > mt0[0, -1]) & (min_theta[0] < mt1[0, 0])
            inter_theta.append(min_theta[:, inter_theta_i])

    inter_theta_i = min_theta[0] > new_min_theta[-1][0, -1]
    inter_theta.append(min_theta[:, inter_theta_i])

    # zipper together values and gaps
    tmp = []
    for j in range(len(new_min_theta)):
        tmp.append(inter_theta[j])
        tmp.append(new_min_theta[j])
    tmp.append(inter_theta[-1])

    min_theta = np.concatenate(tmp, axis=1)

    if i % 10 == 1:
        ax.plot(phi_srt, th_srt, '.')
        ax.plot(min_theta[0], min_theta[1], 'k')

    i_0 = i_1
    d_0 = proj_d_src[i_0]
    i_1 = np.argmax(proj_d_src[i_0:] > d_0 + 2 * surf.dx) + i_0
    i += 1
    if np.concatenate(shadow_record).size != i_0:
        1/0

# take care of last bit
if i_0 < phi_src.size:
    phi_range = phi_src[i_0: ]
    phi_si = np.argsort(phi_range, kind='heapsort')
    phi_srt = phi_range[phi_si]

    digi_i = np.digitize(phi_srt, min_theta[0])
    shadow_inds = np.isin(digi_i, np.where(shadow_i)[0])
    # undo phi sort
    undo_i = np.argsort(phi_si)
    shadow_inds = shadow_inds[undo_i]

    shadow_record.append(shadow_inds)

shadow_record = np.concatenate(shadow_record)

dist_ma = np.ma.masked_array(eta, mask=~distance_mask)

tmp = np.where(distance_mask)
shadow_mask = distance_mask.copy()
u_shadow_record = shadow_record[u_i]
shadow_mask[tmp[0][u_shadow_record], tmp[1][u_shadow_record]] = False

shadow_ma = np.ma.masked_array(eta, mask=~shadow_mask)

fig, ax_g = plt.subplots()
#ax_g.pcolormesh(surface_realization.x_a, surface_realization.y_a, dist_ma.T, cmap=plt.cm.coolwarm)
ax_g.pcolormesh(surface_realization.x_a, surface_realization.y_a, shadow_ma.T, cmap=plt.cm.coolwarm)

#ax_g.plot(shadow_record[0], shadow_record[1], '.')
