Surface scatter with Kirchhoff approximation
============================================


The Kirchhoff approximation (KA) simplifies surface scatter calculations by assuming
the surface does not change the pressure field incident on the surface. The KA
is used here to calculate pressure scattered from a pressure release surface.
While a frequency domain derivation is followed, the calculation performed here
assumes a short time pulse as a transmission signal.

Kirchhoff approximation calculation
-----------------------------------

With a pressure release surface, Green's theorem for reduces the dimenension of
the Helmholtz equation from 3 to 2 dimensions and is expressed

.. math:: \Phi_{sca}(r) = 2 \int_S \frac{\partial}{\partial n} G_s(a) \, G_r(a) \, da,

where :math:`G_s(a)` indicates the Green's function for a source at point
:math:`s` to point :math:`a`. Point :math:`a` denotes a position on the pressure
release surface. An isospeed medium is assumed for the volume, and the Green's
function is convention is chosen as
:math:`G_s(a)=(4 \pi |a-s|)^{-1} \exp(-i k |a-s|)`. The normal derivative of
the Greens function is defined

.. math:: \frac{\partial}{\partial n} G_s(a) = -ik \frac{\hat{n} \cdot (a - s)}{|a-s|} G_s(a).

With these definitions, the scattered pressure is formulated

.. math:: \Phi_{sca}(r) = \frac{-i k}{8 \pi^2} \iint^\infty_{-\infty}
    \frac{n \cdot (a - s)}{|a-s|}
   \frac{\exp\left[-ik (|a-s| + |r-a|)\right]}{|a-s| |r-a|}
   \, dx \, dy,

where the surface integral is performed along the :math:`z=0` plane.

Stationary phase approximation
------------------------------
The scatter integral can be evaluated with using the stationary phase
approximation, which is equvivalent to the high frequency asimptotics of ray
theory. The scatter integral is written in stationary phase form as

.. math:: \Phi_{sca}(r) = -i k \iint^\infty_{-\infty}
    A(\mathbf{x}) \exp\left[-ik \, p(\mathbf{x}) \right] \, dx \, dy,
with all the real and spatially dependent terms are collected into the slowly varingy
amplitude function :math:`A(\mathbf{x})`. The stationary phase approximation gives the
solution as

.. math:: \Phi_{sca}(r) \approx -2 i \pi \sum_{x_0}
          | \textrm{H}(\mathbf{x}_0) | ^{-1/2}
          \exp\left[-ik\,  p(\mathbf{x}_0) + i \pi \, \textrm{sgn}(\textrm{H}(\mathbf{x}_0))) / 4\right]
          A(x_0)
where :math:`\textbf{x}_0` are stationary points of the function
:math:`s(\textbf{x})`.

The stationary phase can also be aplied to only one dimension of the KA
integral, leading to an integral that includes finite frequency effects in one
dimension. The stationary phase approximation assumes negligable scatter from
one of the two surface directions. This out of plane direction can be either
perpendicular to a straight line between the source and receiver, or
pre-determined based on anisotropy of the surface wave spectrum.


.. math:: \Phi_{sca}(r) = \exp[-i \pi / 4] \sqrt{\frac{2 \pi}{k | f''(x) | }}
    \int^\infty_{-\infty}
    \frac{n \cdot (a - s)}{|a-s|}
   \frac{\exp\left[-ik (|a-s| + |r-a|)\right]}{|a-s| |r-a|}
   \, dx

asd


Kirchhoff approximation in time domain
--------------------------------------

A short time pulse is assumed to bound the infinite surface integral.
Window functions are a good way to create a short pulse with efficent
bandwidth, which is important to make this scatter calculation efficent. The
pulse transmission function is introduced as :math:`W(k)`. The Fourier
transform is introduced as :math:`\mathcal{F}`,

.. math:: p_{sca}(r) = \frac{1}{8 \pi^2} \iint^\infty_{-\infty}
    \frac{n \cdot (a - s)}{|a-s|^2 |r-a|}
   \mathcal{F}\left\{-ik W(k) \exp\left[-ik (|a-s| + |r-a|)\right]\right\}
   \, dx \, dy,
where the integral of the Fourier transform is moved inside of the spatial
integrals to only include terms dependent on :math:`k`. The Fourier transform
can be performed analytically,

.. math:: \mathcal{F}\left\{-ik W(k) \exp\left[-ik (|a-s| + |r-a|)\right]\right\}
   = w'(t - \tau),
where :math:`\tau=(|a-s| + |r-a|) c^{-1}` and :math:`w'(t)` denotes a
derivative of :math:`w` with respect to time.


There is an equivalence in the time domain formulation of the KA between the
extent of spatial integration and the delay :math:`\tau` that appears in the
integral. If we assume that all scatter from the surface appears before a
maximum delay, :math:`\tau_{max}`, the integral can be spatially bounded to only
include delays before this delay. This spatially bounded integral is written

.. math:: P_{sca}(r) = -\frac{i k}{8 \pi^2} \iint_D
    \frac{n \cdot (a - s)}{|a-s|}
   \frac{\exp\left[-ik (|a-s| + |r-a|)\right]}{|a-s| |r-a|}
   \, dx \, dy,

where :math:`D` is all positions where :math:`|a-s| + |r-a| < c^{-1}\,  \tau_{max}`.
