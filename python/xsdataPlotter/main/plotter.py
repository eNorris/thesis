import numpy
import matplotlib.pyplot as pyplot

__author__ = 'etnc6d'



smallx = numpy.array([
    2e+07,
    1e+07,
    8e+06,
    6.5e+06,
    5e+06,
    4e+06,
    3e+06,
    2.5e+06,
    2e+06,
    1.66e+06,
    1.33e+06,
    1e+06,
    800000,
    600000,
    400000,
    300000,
    200000,
    100000,
    45000,
    10000
])

bigx = numpy.array([
    2e+07,	1.4e+07,	1.2e+07,	1e+07,	8e+06,	7.5e+06,	7e+06,	6.5e+06,	6e+06,	5.5e+06,	5e+06,
    4.5e+06,	4e+06,	3.5e+06,	3e+06,	2.75e+06,	2.5e+06,	2.35e+06,	2.15e+06,	2e+06,	1.8e+06,
    1.66e+06,	1.57e+06,	1.5e+06,	1.44e+06,	1.33e+06,	1.2e+06,	1e+06,	900000,	800000,	700000,	600000,
    512000,	510000,	450000,	400000,	300000,	260000,	200000,	150000,	100000,	75000,	70000,	60000,	45000,	30000,
    20000,	10000
])

true = numpy.array(
    [[1.000E-02,	1.459E+00,	1.888E+00,	6.329E+00,	0.000E+00,	0.000E+00,	9.676E+00,	8.217E+00],
	[1.500E-02,	8.350E-01,	2.112E+00,	1.647E+00,	0.000E+00,	0.000E+00,	4.594E+00,	3.759E+00],
	[2.000E-02,	5.295E-01,	2.210E+00,	6.296E-01,	0.000E+00,	0.000E+00,	3.369E+00,	2.840E+00],
	[3.000E-02,	2.649E-01,	2.256E+00,	1.611E-01,	0.000E+00,	0.000E+00,	2.682E+00,	2.417E+00],
	[4.000E-02,	1.577E-01,	2.235E+00,	6.106E-02,	0.000E+00,	0.000E+00,	2.454E+00,	2.296E+00],
	[4.500E-02,	1.268E-01,	2.215E+00,	4.100E-02,	0.000E+00,	0.000E+00,	2.383E+00,	2.256E+00],
	[5.000E-02,	1.041E-01,	2.193E+00,	2.872E-02,	0.000E+00,	0.000E+00,	2.326E+00,	2.222E+00],
	[6.000E-02,	7.363E-02,	2.145E+00,	1.555E-02,	0.000E+00,	0.000E+00,	2.234E+00,	2.161E+00],
	[8.000E-02,	4.224E-02,	2.048E+00,	5.891E-03,	0.000E+00,	0.000E+00,	2.096E+00,	2.054E+00],
	[1.000E-01,	2.729E-02,	1.957E+00,	2.782E-03,	0.000E+00,	0.000E+00,	1.987E+00,	1.960E+00],
	[1.500E-01,	1.225E-02,	1.768E+00,	7.210E-04,	0.000E+00,	0.000E+00,	1.781E+00,	1.769E+00],
	[2.000E-01,	6.915E-03,	1.623E+00,	2.815E-04,	0.000E+00,	0.000E+00,	1.630E+00,	1.623E+00],
	[3.000E-01,	3.081E-03,	1.413E+00,	7.833E-05,	0.000E+00,	0.000E+00,	1.416E+00,	1.413E+00],
	[4.000E-01,	1.735E-03,	1.266E+00,	3.328E-05,	0.000E+00,	0.000E+00,	1.268E+00,	1.266E+00],
	[5.000E-01,	1.111E-03,	1.157E+00,	1.786E-05,	0.000E+00,	0.000E+00,	1.158E+00,	1.157E+00],
	[6.000E-01,	7.715E-04,	1.070E+00,	1.109E-05,	0.000E+00,	0.000E+00,	1.071E+00,	1.070E+00],
	[8.000E-01,	4.341E-04,	9.403E-01,	5.591E-06,	0.000E+00,	0.000E+00,	9.407E-01,	9.403E-01],
	[1.000E+00,	2.778E-04,	8.455E-01,	3.476E-06,	0.000E+00,	0.000E+00,	8.458E-01,	8.455E-01],
	[1.022E+00,	2.660E-04,	8.365E-01,	3.144E-06,	0.000E+00,	0.000E+00,	8.368E-01,	8.365E-01],
	[1.250E+00,	1.778E-04,	7.560E-01,	2.137E-06,	1.260E-04,	0.000E+00,	7.563E-01,	7.561E-01],
	[1.330E+00,	1.571E-04,	7.322E-01,	1.950E-06,	2.615E-04,	0.000E+00,	7.326E-01,	7.324E-01],
	[1.500E+00,	1.235E-04,	6.872E-01,	1.555E-06,	7.043E-04,	0.000E+00,	6.880E-01,	6.879E-01],
	[1.660E+00,	1.008E-04,	6.505E-01,	1.305E-06,	1.277E-03,	0.000E+00,	6.519E-01,	6.518E-01],
	[2.000E+00,	6.947E-05,	5.864E-01,	9.844E-07,	2.818E-03,	0.000E+00,	5.893E-01,	5.892E-01],
	[2.044E+00,	6.651E-05,	5.792E-01,	9.528E-07,	3.038E-03,	0.000E+00,	5.823E-01,	5.822E-01],
	[2.500E+00,	4.446E-05,	5.151E-01,	7.119E-07,	5.424E-03,	2.791E-05,	5.206E-01,	5.206E-01],
	[3.000E+00,	3.087E-05,	4.613E-01,	5.550E-07,	8.082E-03,	1.614E-04,	4.696E-01,	4.695E-01],
	[4.000E+00,	1.737E-05,	3.848E-01,	3.831E-07,	1.313E-02,	6.590E-04,	3.986E-01,	3.986E-01],
	[5.000E+00,	1.112E-05,	3.323E-01,	2.916E-07,	1.763E-02,	1.313E-03,	3.513E-01,	3.512E-01],
	[6.000E+00,	7.719E-06,	2.937E-01,	2.351E-07,	2.167E-02,	2.017E-03,	3.174E-01,	3.174E-01],
	[6.500E+00,	6.577E-06,	2.780E-01,	2.143E-07,	2.352E-02,	2.370E-03,	3.039E-01,	3.039E-01],
	[7.000E+00,	5.671E-06,	2.640E-01,	1.968E-07,	2.528E-02,	2.719E-03,	2.920E-01,	2.920E-01],
	[8.000E+00,	4.342E-06,	2.403E-01,	1.691E-07,	2.854E-02,	3.401E-03,	2.722E-01,	2.722E-01],
	[9.000E+00,	3.431E-06,	2.209E-01,	1.482E-07,	3.150E-02,	4.055E-03,	2.565E-01,	2.565E-01],
	[1.000E+01,	2.779E-06,	2.046E-01,	1.319E-07,	3.420E-02,	4.679E-03,	2.435E-01,	2.435E-01],
	[1.100E+01,	2.296E-06,	1.909E-01,	1.189E-07,	3.665E-02,	5.271E-03,	2.328E-01,	2.328E-01],
	[1.200E+01,	1.930E-06,	1.790E-01,	1.081E-07,	3.890E-02,	5.830E-03,	2.237E-01,	2.237E-01],
	[1.300E+01,	1.644E-06,	1.686E-01,	9.916E-08,	4.100E-02,	6.359E-03,	2.160E-01,	2.160E-01],
	[1.400E+01,	1.418E-06,	1.595E-01,	9.156E-08,	4.296E-02,	6.865E-03,	2.093E-01,	2.093E-01],
	[1.500E+01,	1.235E-06,	1.515E-01,	8.505E-08,	4.479E-02,	7.344E-03,	2.036E-01,	2.036E-01],
	[1.600E+01,	1.085E-06,	1.442E-01,	7.939E-08,	4.651E-02,	7.802E-03,	1.985E-01,	1.985E-01],
	[1.800E+01,	8.576E-07,	1.318E-01,	7.007E-08,	4.968E-02,	8.655E-03,	1.901E-01,	1.901E-01],
	[2.000E+01,	6.947E-07,	1.216E-01,	6.271E-08,	5.253E-02,	9.436E-03,	1.836E-01,	1.836E-01]])

te = true[:, 0]
te *= 1E6
tcoh = true[:, 1]
tinc = true[:, 2]
tpe = true[:, 3]
tpairn = true[:, 4]
tpaire = true[:, 5]
tpair = tpairn + tpaire
ttot = true[:, 6]
ttotwocoh = true[:, 7]

for e in smallx:
    print(e/1000000)

smallv = numpy.array([
[ 0.219909 ,  0.046296 ,  1.88556e-06 ,  0.173611 ,  1.04917e-07 ],
[ 0.256612 ,  0.0353548 ,  3.53943e-06 ,  0.221254 ,  1.49747e-07 ],
[ 0.286715 ,  0.0288468 ,  5.43926e-06 ,  0.257862 ,  1.90839e-07 ],
[ 0.325858 ,  0.0223033 ,  8.74005e-06 ,  0.303546 ,  2.50884e-07 ],
[ 0.373294 ,  0.0162716 ,  1.41577e-05 ,  0.357008 ,  3.35161e-07 ],
[ 0.431982 ,  0.0108714 ,  2.37255e-05 ,  0.421087 ,  4.63127e-07 ],
[ 0.493412 ,  0.00677968 ,  3.76494e-05 ,  0.486594 ,  6.29941e-07 ],
[ 0.553039 ,  0.00405447 ,  5.66282e-05 ,  0.548927 ,  8.39846e-07 ],
[ 0.618849 ,  0.00199198 ,  8.50664e-05 ,  0.616771 ,  1.13733e-06 ],
[ 0.690404 ,  0.000695119 ,  0.000128221 ,  0.689579 ,  1.59797e-06 ],
[ 0.787117 ,  6.44206e-05 ,  0.000213978 ,  0.786836 ,  2.62972e-06 ],
[ 0.891595 ,  0 ,  0.000353846 ,  0.891236 ,  4.44164e-06 ],
[ 1.00408 ,  0 ,  0.000592737 ,  1.00347 ,  7.97945e-06 ],
[ 1.16721 ,  0 ,  0.00120084 ,  1.16599 ,  1.99754e-05 ],
[ 1.34041 ,  0 ,  0.00236812 ,  1.33799 ,  5.24373e-05 ],
[ 1.52119 ,  0 ,  0.00479035 ,  1.51624 ,  0.000158053 ],
[ 1.80931 ,  0 ,  0.0150122 ,  1.79321 ,  0.00108521 ],
[ 2.10879 ,  0 ,  0.0490544 ,  2.05122 ,  0.00852082 ],
[ 2.74115 ,  0 ,  0.26378 ,  2.22882 ,  0.24855 ]
])

bigv = numpy.array([
    [ 0.197392 ,  0.0547145 ,  1.09255e-06 ,  0.142676 ,  7.89312e-08 ],
[ 0.216258 ,  0.0470482 ,  1.69931e-06 ,  0.169208 ,  1.00319e-07 ],
[ 0.233557 ,  0.0414947 ,  2.39463e-06 ,  0.19206 ,  1.20857e-07 ],
[ 0.256612 ,  0.0353548 ,  3.53943e-06 ,  0.221254 ,  1.49747e-07 ],
[ 0.275995 ,  0.0309424 ,  4.6836e-06 ,  0.245047 ,  1.7552e-07 ],
[ 0.285771 ,  0.0289707 ,  5.35318e-06 ,  0.256795 ,  1.89322e-07 ],
[ 0.29693 ,  0.0269066 ,  6.17748e-06 ,  0.270017 ,  2.05592e-07 ],
[ 0.309678 ,  0.0247457 ,  7.20826e-06 ,  0.284925 ,  2.24697e-07 ],
[ 0.324106 ,  0.0224724 ,  8.52033e-06 ,  0.301625 ,  2.47588e-07 ],
[ 0.341046 ,  0.0200977 ,  1.02271e-05 ,  0.320938 ,  2.75884e-07 ],
[ 0.360862 ,  0.0176482 ,  1.2504e-05 ,  0.343201 ,  3.11084e-07 ],
[ 0.384415 ,  0.0150402 ,  1.56369e-05 ,  0.369358 ,  3.56699e-07 ],
[ 0.412998 ,  0.0123562 ,  2.01181e-05 ,  0.400621 ,  4.17525e-07 ],
[ 0.448428 ,  0.00958529 ,  2.68503e-05 ,  0.438815 ,  5.02628e-07 ],
[ 0.480035 ,  0.00750521 ,  3.40807e-05 ,  0.472495 ,  5.89042e-07 ],
[ 0.505624 ,  0.00611733 ,  4.09073e-05 ,  0.499465 ,  6.67278e-07 ],
[ 0.528488 ,  0.00502547 ,  4.78276e-05 ,  0.523414 ,  7.44589e-07 ],
[ 0.551071 ,  0.00408388 ,  5.56506e-05 ,  0.546931 ,  8.29642e-07 ],
[ 0.576464 ,  0.00318755 ,  6.53599e-05 ,  0.573211 ,  9.33896e-07 ],
[ 0.605277 ,  0.00232259 ,  7.81428e-05 ,  0.602876 ,  1.06628e-06 ],
[ 0.636508 ,  0.00156178 ,  9.40757e-05 ,  0.634851 ,  1.22979e-06 ],
[ 0.660599 ,  0.00110241 ,  0.000107796 ,  0.659387 ,  1.37286e-06 ],
[ 0.678564 ,  0.000817278 ,  0.00011927 ,  0.677626 ,  1.4958e-06 ],
[ 0.694019 ,  0.000610975 ,  0.000130026 ,  0.693277 ,  1.6146e-06 ],
[ 0.716251 ,  0.00038253 ,  0.000146759 ,  0.71572 ,  1.80597e-06 ],
[ 0.751201 ,  0.000150925 ,  0.000176235 ,  0.750871 ,  2.15733e-06 ],
[ 0.80738 ,  1.56192e-05 ,  0.000235271 ,  0.807126 ,  2.89621e-06 ],
[ 0.866547 ,  0 ,  0.000312529 ,  0.866231 ,  3.89132e-06 ],
[ 0.914 ,  0 ,  0.000390805 ,  0.913604 ,  4.93392e-06 ],
[ 0.969142 ,  0 ,  0.000502686 ,  0.968632 ,  6.54048e-06 ],
[ 1.03434 ,  0 ,  0.000670743 ,  1.03366 ,  9.22596e-06 ],
[ 1.10725 ,  0 ,  0.000917009 ,  1.10632 ,  1.37138e-05 ],
[ 1.14651 ,  0 ,  0.00107415 ,  1.14541 ,  1.68631e-05 ],
[ 1.17767 ,  0 ,  0.0012252 ,  1.17643 ,  2.0184e-05 ],
[ 1.23751 ,  0 ,  0.00156136 ,  1.23592 ,  2.82891e-05 ],
[ 1.34041 ,  0 ,  0.00236812 ,  1.33799 ,  5.24373e-05 ],
[ 1.4518 ,  0 ,  0.00360304 ,  1.4481 ,  9.8553e-05 ],
[ 1.55904 ,  0 ,  0.00543794 ,  1.55341 ,  0.000190506 ],
[ 1.70454 ,  0 ,  0.0094241 ,  1.69465 ,  0.000466589 ],
[ 1.88364 ,  0 ,  0.018977 ,  1.86314 ,  0.00152412 ],
[ 2.04587 ,  0 ,  0.0355523 ,  2.00594 ,  0.00437605 ],
[ 2.14051 ,  0 ,  0.0516378 ,  2.08066 ,  0.00820583 ],
[ 2.19263 ,  0 ,  0.0636135 ,  2.11715 ,  0.0118749 ],
[ 2.29481 ,  0 ,  0.0951254 ,  2.17518 ,  0.0244974 ],
[ 2.48864 ,  0 ,  0.176312 ,  2.23574 ,  0.0765895 ],
[ 2.90559 ,  0 ,  0.361205 ,  2.24445 ,  0.29993 ],
[ 4.60798 ,  0 ,  0.800614 ,  2.12299 ,  1.68438 ]
])

tot = []
pair = []
coh = []
inc = []
pe = []

#usex = smallx
#usetot = smallv[:,0]
#usepair = smallv[:,1]
#usecoh = smallv[:,2]
#useinc = smallv[:,3]
#usepe = smallv[:,4]

usex = bigx
usetot = bigv[:,0]
usepair = bigv[:,1]
usecoh = bigv[:,2]
useinc = bigv[:,3]
usepe = bigv[:,4]

x = [usex[0]]
for k in usex[1:-1]:
    x.append(k)
    x.append(k)
x.append(smallx[-1])

for k in usetot:
    tot.append(k)
    tot.append(k)

for k in usepair:
    pair.append(k)
    pair.append(k)

for k in usecoh:
    coh.append(k)
    coh.append(k)

for k in useinc:
    inc.append(k)
    inc.append(k)

for k in usepe:
    pe.append(k)
    pe.append(k)

pyplot.figure()

pyplot.loglog(x, tot, "k", x, pair, "b", x, coh, "g", x, inc, "r", x, pe, "c",
              te, ttot, "k--", te, tpair, "b--", te, tcoh, "g--", te, tinc, "r--", te, tpe, "c--")
pyplot.title("Comparison of Cross Section Data (47 Group)")
pyplot.xlabel("Energy (eV)")
pyplot.ylabel("Cross Section (barn)")
pyplot.legend(["Total", "Pair Prod.", "Coherent", "Incoherent", "Photo Elec.", "Total", "Pair Prod.", "Coherent", "Incoherent", "Photo Elec."])


pyplot.show()