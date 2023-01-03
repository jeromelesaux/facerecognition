package algorithm

import (
	"math"
)

type EigenvalueDecomposition struct {
	N           int         // Row and column dimension (square matrix).
	IsSymmetric bool        //  Symmetry flag.
	D           []float64   // Arrays for internal storage of eigenvalues.
	E           []float64   // Array for internal storage of eigenvectors.
	V           [][]float64 // Array for internal storage of eigenvectors.
	H           [][]float64 //  Array for internal storage of nonsymmetric Hessenberg form.
	Ort         []float64   // Working storage for nonsymmetric algorithm.
}

var (
	cdivr = 0.0
	cdivi = 0.0
)

// This is derived from the Algol procedures tred2 by
// Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
// Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
// Fortran subroutine in EISPACK.
func (e *EigenvalueDecomposition) Tred2() {
	//logger.Log("starting tred2")
	for j := 0; j < e.N; j++ {
		e.D[j] = e.V[e.N-1][j]
	}

	// Householder reduction to tridiagonal form.

	for i := e.N - 1; i > 0; i-- {
		// Scale to avoid under/overflow.

		scale := 0.0
		h := 0.0

		for k := 0; k < i; k++ {
			scale = scale + math.Abs(e.D[k])
		}
		if scale == 0.0 {
			e.E[i] = e.D[i-1]
			for j := 0; j < i; j++ {
				e.D[j] = e.V[i-1][j]
				e.V[i][j] = 0.0
				e.V[j][i] = 0.0
			}
		} else {

			// Generate Householder vector.

			for k := 0; k < i; k++ {
				e.D[k] /= scale
				h += e.D[k] * e.D[k]
			}

			f := e.D[i-1]
			g := math.Sqrt(h)
			if f > 0 {
				g = -g
			}
			e.E[i] = scale * g
			h = h - (f * g)
			e.D[i-1] = f - g
			for j := 0; j < i; j++ {
				e.E[j] = 0.0
			}

			// Apply similarity transformation to remaining columns.

			for j := 0; j < i; j++ {
				f = e.D[j]
				e.V[j][i] = f
				g = e.E[j] + (e.V[j][j] * f)
				for k := j + 1; k <= i-1; k++ {
					g += e.V[k][j] * e.D[k]
					e.E[k] += e.V[k][j] * f
				}
				e.E[j] = g
			}
			f = 0.0
			for j := 0; j < i; j++ {
				e.E[j] /= h
				f += e.E[j] * e.D[j]
			}

			hh := f / (h + h)
			for j := 0; j < i; j++ {
				e.E[j] -= hh * e.D[j]
			}
			for j := 0; j < i; j++ {
				f = e.D[j]
				g = e.E[j]
				for k := j; k <= i-1; k++ {
					e.V[k][j] -= ((f * e.E[k]) + (g * e.D[k]))
				}
				e.D[j] = e.V[i-1][j]
				e.V[i][j] = 0.0
			}
		}
		e.D[i] = h
	}

	// Accumulate transformations.

	for i := 0; i < e.N-1; i++ {
		e.V[e.N-1][i] = e.V[i][i]
		e.V[i][i] = 1.0
		h := e.D[i+1]
		if h != 0.0 {
			for k := 0; k <= i; k++ {
				e.D[k] = e.V[k][i+1] / h
			}
			for j := 0; j <= i; j++ {
				g := 0.0
				for k := 0; k <= i; k++ {
					g += e.V[k][i+1] * e.V[k][j]
				}
				for k := 0; k <= i; k++ {
					e.V[k][j] -= g * e.D[k]
				}
			}
		}
		for k := 0; k <= i; k++ {
			e.V[k][i+1] = 0.0
		}
	}
	for j := 0; j < e.N; j++ {
		e.D[j] = e.V[e.N-1][j]
		e.V[e.N-1][j] = 0.0
	}
	//logger.Log("dimensions of e.V :" + strconv.Itoa(len(e.V)))
	e.V[e.N-1][e.N-1] = 1.0
	e.E[0] = 0.0
	//logger.Log("tred2 ended")
}

// Symmetric tridiagonal QL algorithm.

func (e *EigenvalueDecomposition) Tql2() {
	//logger.Log("starting tql2")
	//  This is derived from the Algol procedures tql2, by
	//  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
	//  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
	//  Fortran subroutine in EISPACK.

	for i := 1; i < e.N; i++ {
		//fmt.Printf("i:%d,e[i-1]:%f,e[i]:%f\n",i,e.E[i-1],e.E[i])
		e.E[i-1] = e.E[i]
	}
	e.E[e.N-1] = 0.0

	f := 0.0
	tst1 := 0.0
	eps := math.Pow(2.0, -52.0)
	for l := 0; l < e.N; l++ {

		// Find small subdiagonal element

		tst1 = math.Max(tst1, math.Abs(e.D[l])+math.Abs(e.E[l]))
		m := l

		for m < e.N {
			if math.Abs(e.E[m]) <= eps*tst1 {
				break
			}
			m++
		}

		// If m == l, e.D[l] is an eigenvalue,
		// otherwise, iterate.
		if m > l {
			iter := 0
			for {
				// Check for convergence.
				//if math.Abs(e.E[l]) <= (eps * tst1) {
				//	break
				//
				//}

				iter = iter + 1 // (Could check iteration count here.)

				// Compute implicit shift

				g := e.D[l]

				p := (e.D[l+1] - g) / (2.0 * e.E[l])
				r := math.Hypot(p, 1.0)
				if p < 0 {
					r = -r
				}
				e.D[l] = e.E[l] / (p + r)
				e.D[l+1] = e.E[l] * (p + r)
				dl1 := e.D[l+1]
				h := g - e.D[l]

				for i := l + 2; i < e.N; i++ {
					e.D[i] -= h
				}
				f = f + h

				// Implicit QL transformation.

				p = e.D[m]
				c := 1.0
				c2 := 1.0
				c3 := 1.0
				el1 := e.E[l+1]
				s := 0.0
				s2 := 0.0
				for i := (m - 1); i >= l; i-- {
					c3 = c2
					c2 = c
					s2 = s
					g = c * e.E[i]
					h = c * p
					r = math.Hypot(p, e.E[i])

					e.E[i+1] = s * r

					s = e.E[i] / r

					c = p / r
					p = c*e.D[i] - s*g
					e.D[i+1] = h + s*(c*g+s*e.D[i])
					// Accumulate transformation.

					for k := 0; k < e.N; k++ {
						h = e.V[k][i+1]
						e.V[k][i+1] = s*e.V[k][i] + c*h
						e.V[k][i] = c*e.V[k][i] - s*h
					}
				}
				p = -s * s2 * c3 * el1 * e.E[l] / dl1
				e.E[l] = s * p
				e.D[l] = c * p

				// Check for convergence.
				if math.Abs(e.E[l]) <= (eps * tst1) {
					break

				}

			}
		}
		e.D[l] += f
		e.E[l] = 0.0

		// Sort eigenvalues and corresponding vectors.
	}
	for i := 0; i < e.N-1; i++ {
		k := i
		p := e.D[i]
		for j := i + 1; j < e.N; j++ {
			if e.D[j] < p {
				k = j
				p = e.D[j]
			}
		}
		if k != i {
			e.D[k] = e.D[i]
			e.D[i] = p
			for j := 0; j < e.N; j++ {
				pt := e.V[j][i]
				e.V[j][i] = e.V[j][k]
				e.V[j][k] = pt
			}
		}
	}

	//logger.Log("tql2 ended")
}

// Nonsymmetric reduction to Hessenberg form.
func (e *EigenvalueDecomposition) Orthes() {
	//logger.Log("others starting")
	//  This is derived from the Algol procedures orthes and ortran,
	//  by Martin and Wilkinson, Handbook for Auto. Comp.,
	//  Vol.ii-Linear Algebra, and the corresponding
	//  Fortran subroutines in EISPACK.
	low := 0
	high := e.N - 1
	for m := low + 1; m <= high-1; m++ {

		// Scale column.

		scale := 0.0
		for i := m; i <= high; i++ {
			scale = scale + math.Abs(e.H[i][m-1])
		}
		if scale != 0.0 {
			// Compute Householder transformation.
			h := 0.0
			for i := high; i >= m; i-- {
				e.Ort[i] = e.H[i][m-1] / scale
				h += e.Ort[i] * e.Ort[i]
			}
			g := math.Sqrt(h)
			if e.Ort[m] > 0 {
				g = -g
			}
			h = h - e.Ort[m]*g
			e.Ort[m] = e.Ort[m] - g

			// Apply Householder similarity transformation
			// H = (I-u*u'/h)*H*(I-u*u')/h)
			for j := m; j < e.N; j++ {
				f := 0.0
				for i := high; i >= m; i-- {
					f += e.Ort[i] * e.H[i][j]
				}
				f = f / h
				for i := m; i <= high; i++ {
					e.H[i][j] -= f * e.Ort[i]
				}
			}

			for i := 0; i <= high; i++ {
				f := 0.0
				for j := high; j >= m; j-- {
					f += e.Ort[j] * e.H[i][j]
				}
				f = f / h
				for j := m; j <= high; j++ {
					e.H[i][j] -= f * e.Ort[j]
				}
			}
			e.Ort[m] = scale * e.Ort[m]
			e.H[m][m-1] = scale * g
		}
	}

	// Accumulate transformations (Algol's ortran).

	for i := 0; i < e.N; i++ {
		for j := 0; j < e.N; j++ {
			if i == j {
				e.V[i][j] = 1.0
			} else {
				e.V[i][j] = .0
			}
		}
	}

	for m := high - 1; m >= low+1; m-- {
		if e.H[m][m-1] != 0.0 {
			for i := m + 1; i <= high; i++ {
				e.Ort[i] = e.H[i][m-1]
			}
			for j := m; j <= high; j++ {
				g := 0.0
				for i := m; i <= high; i++ {
					g += e.Ort[i] * e.V[i][j]
				}
				// Double division avoids possible underflow
				g = (g / e.Ort[m]) / e.H[m][m-1]
				for i := m; i <= high; i++ {
					e.V[i][j] += g * e.Ort[i]
				}
			}
		}
	}
	//logger.Log("others ended")
}

func (e *EigenvalueDecomposition) Cdiv(xr, xi, yr, yi float64) {
	var r, d float64
	if math.Abs(yr) > math.Abs(yi) {
		r = yi / yr
		d = yr + r*yi
		cdivr = (xr + r*xi) / d
		cdivi = (xi - r*xr) / d
	} else {
		r = yr / yi
		d = yi + r*yr
		cdivr = (r*xr + xi) / d
		cdivi = (r*xi - xr) / d
	}
}

// Nonsymmetric reduction from Hessenberg to real Schur form.

func (e *EigenvalueDecomposition) Hqr2() {
	//  This is derived from the Algol procedure hqr2,
	//  by Martin and Wilkinson, Handbook for Auto. Comp.,
	//  Vol.ii-Linear Algebra, and the corresponding
	//  Fortran subroutine in EISPACK.

	// Initialize
	nn := e.N
	n := nn - 1
	low := 0
	high := nn - 1
	eps := math.Pow(2.0, -52.0)
	exshift := 0.0
	p := 0.
	q := 0.
	r := 0.
	s := 0.
	z := 0.
	var t, w, x, y float64

	// Store roots isolated by balanc and compute matrix norm

	norm := 0.0
	for i := 0; i < nn; i++ {
		if (i < low) || (i > high) {
			e.D[i] = e.H[i][i]
			e.E[i] = 0.0
		}
		for j := maxInt(i-1, 0); j < nn; j++ {
			norm = norm + math.Abs(e.H[i][j])
		}
	}
	// Outer loop over eigenvalue index
	iter := 0
	for n >= low {
		// Look for single small sub-diagonal element
		l := n
		for l > low {
			s = math.Abs(e.H[l-1][l-1]) + math.Abs(e.H[l][l])
			if s == 0.0 {
				s = norm
			}
			if math.Abs(e.H[l][l-1]) < eps*s {
				break
			}
			l--
		}

		// Check for convergence
		// One root found

		if l == n {
			e.H[n][n] = e.H[n][n] + exshift
			e.D[n] = e.H[n][n]
			e.E[n] = 0.0
			n--
			iter = 0
			// Two roots found
		} else if l == n-1 {
			w = e.H[n][n-1] * e.H[n-1][n]
			p = (e.H[n-1][n-1] - e.H[n][n]) / 2.0
			q = p*p + w
			z = math.Sqrt(math.Abs(q))
			e.H[n][n] = e.H[n][n] + exshift
			e.H[n-1][n-1] = e.H[n-1][n-1] + exshift
			x = e.H[n][n]

			// Real pair
			if q >= 0 {
				if p >= 0 {
					z = p + z
				} else {
					z = p - z
				}
				e.D[n-1] = x + z
				e.D[n] = e.D[n-1]
				if z != 0.0 {
					e.D[n] = x - w/z
				}
				e.E[n-1] = 0.0
				e.E[n] = 0.0
				x = e.H[n][n-1]
				s = math.Abs(x) + math.Abs(z)
				p = x / s
				q = z / s
				r = math.Sqrt(p*p + q*q)
				p = p / r
				q = q / r
				// Row modification
				for j := n - 1; j < nn; j++ {
					z = e.H[n-1][j]
					e.H[n-1][j] = q*z + p*e.H[n][j]
					e.H[n][j] = q*e.H[n][j] - p*z
				}
				// Column modification
				for i := 0; i <= n; i++ {
					z = e.H[i][n-1]
					e.H[i][n-1] = q*z + p*e.H[i][n]
					e.H[i][n] = q*e.H[i][n] - p*z
				}
				// Accumulate transformations
				for i := low; i <= high; i++ {
					z = e.V[i][n-1]
					e.V[i][n-1] = q*z + p*e.V[i][n]
					e.V[i][n] = q*e.V[i][n] - p*z
				}
				// Complex pair
			} else {
				e.D[n-1] = x + p
				e.D[n] = x + p
				e.E[n-1] = z
				e.E[n] = -z
			}
			n = n - 2
			iter = 0
			// No convergence yet
		} else {
			// Form shift
			x = e.H[n][n]
			y = 0.0
			w = 0.0
			if l < n {
				y = e.H[n-1][n-1]
				w = e.H[n][n-1] * e.H[n-1][n]
			}
			// Wilkinson's original ad hoc shift
			if iter == 10 {
				exshift += x
				for i := low; i <= e.N; i++ {
					e.H[i][i] -= x
				}
				s = math.Abs(e.H[n][n-1]) + math.Abs(e.H[n-1][n-2])
				y = 0.75 * s
				x = y
				w = -0.4375 * s * s
			}
			// MATLAB's new ad hoc shift
			if iter == 30 {
				s = (y - x) / 2.0
				s = s*s + w
				if s > 0 {
					s = math.Sqrt(s)
					if y < x {
						s = -s
					}
					s = x - w/((y-x)/2.0+s)
					for i := low; i <= e.N; i++ {
						e.H[i][i] -= s
					}
					exshift += s
					w = 0.964
					y = w
					x = y
				}
			}
			iter = iter + 1 // (Could check iteration count here.)
			// Look for two consecutive small sub-diagonal elements
			m := n - 2
			for m >= l {
				z = e.H[m][m]
				r = x - z
				s = y - z
				p = (r*s-w)/e.H[m+1][m] + e.H[m][m+1]
				q = e.H[m+1][m+1] - z - r - s
				r = e.H[m+2][m+1]
				s = math.Abs(p) + math.Abs(q) + math.Abs(r)
				p = p / s
				q = q / s
				r = r / s
				if m == l {
					break
				}
				if math.Abs(e.H[m][m-1])*(math.Abs(q)+math.Abs(r)) <
					eps*(math.Abs(p)*(math.Abs(e.H[m-1][m-1])+math.Abs(z)+
						math.Abs(e.H[m+1][m+1]))) {
					break
				}
				m--
			}
			for i := m + 2; i <= n; i++ {
				e.H[i][i-2] = 0.0
				if i > m+2 {
					e.H[i][i-3] = 0.0
				}
			}
			// Double QR step involving rows l:n and columns m:n
			for k := m; k <= n-1; k++ {
				notlast := (k != n-1)
				if k != m {
					p = e.H[k][k-1]
					q = e.H[k+1][k-1]
					if notlast {
						r = e.H[k+2][k-1]
					} else {
						r = 0.0
					}

					x = math.Abs(p) + math.Abs(q) + math.Abs(r)
					if x == 0.0 {
						continue
					}
					p = p / x
					q = q / x
					r = r / x
				}

				s = math.Sqrt(p*p + q*q + r*r)
				if p < 0 {
					s = -s
				}
				if s != 0 {
					if k != m {
						e.H[k][k-1] = -s * x
					} else if l != m {
						e.H[k][k-1] = -e.H[k][k-1]
					}
					p = p + s
					x = p / s
					y = q / s
					z = r / s
					q = q / p
					r = r / p
					// Row modification
					for j := k; j < nn; j++ {
						p = e.H[k][j] + q*e.H[k+1][j]
						if notlast {
							p = p + r*e.H[k+2][j]
							e.H[k+2][j] = e.H[k+2][j] - p*z
						}
						e.H[k][j] = e.H[k][j] - p*x
						e.H[k+1][j] = e.H[k+1][j] - p*y
					}
					// Column modification
					for i := 0; i <= minInt(n, k+3); i++ {
						p = x*e.H[i][k] + y*e.H[i][k+1]
						if notlast {
							p = p + z*e.H[i][k+2]
							e.H[i][k+2] = e.H[i][k+2] - p*r
						}
						e.H[i][k] = e.H[i][k] - p
						e.H[i][k+1] = e.H[i][k+1] - p*q
					}
					// Accumulate transformations
					for i := low; i <= high; i++ {
						p = x*e.V[i][k] + y*e.V[i][k+1]
						if notlast {
							p = p + z*e.V[i][k+2]
							e.V[i][k+2] = e.V[i][k+2] - p*r
						}
						e.V[i][k] = e.V[i][k] - p
						e.V[i][k+1] = e.V[i][k+1] - p*q
					}
				} // (s != 0)
			} // k loop
		} // check convergence
	} // while (n >= low)

	// Backsubstitute to find vectors of upper triangular form

	if norm == 0.0 {
		return
	}

	for n = nn - 1; n >= 0; n-- {
		p = e.D[n]
		q = e.E[n]
		// Real vector
		if q == 0 {
			l := n
			e.H[n][n] = 1.0
			for i := n - 1; i >= 0; i-- {
				w = e.H[i][i] - p
				r = 0.0
				for j := l; j <= n; j++ {
					r = r + e.H[i][j]*e.H[j][n]
				}
				if e.E[i] < 0.0 {
					z = w
					s = r
				} else {
					l = i
					if e.E[i] == 0.0 {
						if w != 0.0 {
							e.H[i][n] = -r / w
						} else {
							e.H[i][n] = -r / (eps * norm)
						}
						// Solve real equations
					} else {
						x = e.H[i][i+1]
						y = e.H[i+1][i]
						q = (e.D[i]-p)*(e.D[i]-p) + e.E[i]*e.E[i]
						t = (x*s - z*r) / q
						e.H[i][n] = t
						if math.Abs(x) > math.Abs(z) {
							e.H[i+1][n] = (-r - w*t) / x
						} else {
							e.H[i+1][n] = (-s - y*t) / z
						}
					}
					// Overflow control
					t = math.Abs(e.H[i][n])
					if (eps*t)*t > 1 {
						for j := i; j <= n; j++ {
							e.H[j][n] = e.H[j][n] / t
						}
					}
				}
			}
			// Complex vector
		} else if q < 0 {
			l := n - 1
			// Last vector component imaginary so matrix is triangular

			if math.Abs(e.H[n][n-1]) > math.Abs(e.H[n-1][n]) {
				e.H[n-1][n-1] = q / e.H[n][n-1]
				e.H[n-1][n] = -(e.H[n][n] - p) / e.H[n][n-1]
			} else {
				e.Cdiv(0.0, -e.H[n-1][n], e.H[n-1][n-1]-p, q)
				e.H[n-1][n-1] = cdivr
				e.H[n-1][n] = cdivi
			}
			e.H[n][n-1] = 0.0
			e.H[n][n] = 1.0
			for i := n - 2; i >= 0; i-- {
				ra := 0.0
				sa := 0.0
				vr := 0.0
				vi := 0.0
				for j := l; j <= n; j++ {
					ra = ra + e.H[i][j]*e.H[j][n-1]
					sa = sa + e.H[i][j]*e.H[j][n]
				}
				w = e.H[i][i] - p
				if e.E[i] < 0.0 {
					z = w
					r = ra
					s = sa
				} else {
					l = i
					if e.E[i] == 0 {
						e.Cdiv(-ra, -sa, w, q)
						e.H[i][n-1] = cdivr
						e.H[i][n] = cdivi
					} else {
						// Solve complex equations
						x = e.H[i][i+1]
						y = e.H[i+1][i]
						vr = (e.D[i]-p)*(e.D[i]-p) + e.E[i]*e.E[i] - q*q
						vi = (e.D[i] - p) * 2.0 * q
						if vr == 0.0 && vi == 0.0 {
							vr = eps * norm * (math.Abs(w) + math.Abs(q) +
								math.Abs(x) + math.Abs(y) + math.Abs(z))
						}
						e.Cdiv(x*r-z*ra+q*sa, x*s-z*sa-q*ra, vr, vi)
						e.H[i][n-1] = cdivr
						e.H[i][n] = cdivi
						if math.Abs(x) > (math.Abs(z) + math.Abs(q)) {
							e.H[i+1][n-1] = (-ra - w*e.H[i][n-1] + q*e.H[i][n]) / x
							e.H[i+1][n] = (-sa - w*e.H[i][n] - q*e.H[i][n-1]) / x
						} else {
							e.Cdiv(-r-y*e.H[i][n-1], -s-y*e.H[i][n], z, q)
							e.H[i+1][n-1] = cdivr
							e.H[i+1][n] = cdivi
						}
					}

					// Overflow control

					t = math.Max(math.Abs(e.H[i][n-1]), math.Abs(e.H[i][n]))
					if (eps*t)*t > 1 {
						for j := i; j <= n; j++ {
							e.H[j][n-1] = e.H[j][n-1] / t
							e.H[j][n] = e.H[j][n] / t
						}
					}
				}
			}
		}
	}
	// Vectors of isolated roots
	for i := 0; i < nn; i++ {
		if (i < low) || (i > high) {
			for j := i; j < nn; j++ {
				e.V[i][j] = e.H[i][j]
			}
		}
	}
	// Back transformation to get eigenvectors of original matrix
	for j := nn - 1; j >= low; j-- {
		for i := low; i <= high; i++ {
			z = 0.0
			for k := low; k <= minInt(j, high); k++ {
				z = z + e.V[i][k]*e.H[k][j]
			}
			e.V[i][j] = z
		}
	}
}

/* ------------------------
   Constructor
 * ------------------------ */

/** Check for symmetry, then construct the eigenvalue decomposition
    Structure to access D and V.
@param Arg    Square matrix
*/

func NewEigenvalueDecomposition(matrix *Matrix) *EigenvalueDecomposition {
	//logger.Log(matrix.Tostring())
	e := &EigenvalueDecomposition{}
	A := matrix.A
	e.N = matrix.ColumnsDimension()
	e.V = make([][]float64, e.N)
	for i := 0; i < e.N; i++ {
		e.V[i] = make([]float64, e.N)
	}
	e.D = make([]float64, e.N)
	e.E = make([]float64, e.N)

	e.IsSymmetric = true
	for j := 0; (j < e.N) && e.IsSymmetric; j++ {
		for i := 0; (i < e.N) && e.IsSymmetric; i++ {
			e.IsSymmetric = (A[i][j] == A[j][i])
		}
	}

	if e.IsSymmetric {
		for i := 0; i < e.N; i++ {
			for j := 0; j < e.N; j++ {
				e.V[i][j] = A[i][j]
			}
		}

		// Tridiagonalize.
		e.Tred2()

		// Diagonalize.
		e.Tql2()

	} else {
		e.H = make([][]float64, e.N)
		for i := 0; i < e.N; i++ {
			e.H[i] = make([]float64, e.N)
		}
		e.Ort = make([]float64, e.N)

		for j := 0; j < e.N; j++ {
			for i := 0; i < e.N; i++ {
				e.H[i][j] = A[i][j]
			}
		}

		// Reduce to Hessenberg form.
		e.Orthes()

		// Reduce Hessenberg to real Schur form.
		e.Hqr2()
	}
	return e
}

/* ------------------------
   Public Methods
 * ------------------------ */

/** Return the eigenvector matrix
@return     V
*/

func (e *EigenvalueDecomposition) GetV() *Matrix {
	mat, _ := NewMatrixWithMatrix(e.V, e.N, e.N)
	return mat
}

/** Return the real parts of the eigenvalues
@return     real(diag(D))
*/

func (e *EigenvalueDecomposition) GetRealEigenvalues() []float64 {
	return e.D
}

/** Return the imaginary parts of the eigenvalues
@return     imag(diag(D))
*/

func (e *EigenvalueDecomposition) GetImagEigenvalues() []float64 {
	return e.E
}

/** Return the block diagonal eigenvalue matrix
@return     D
*/

func (e *EigenvalueDecomposition) GetD() *Matrix {

	X := NewMatrix(e.N, e.N)

	for i := 0; i < e.N; i++ {
		for j := 0; j < e.N; j++ {
			X.A[i][j] = 0.0
		}
		X.A[i][i] = e.D[i]
		if e.E[i] > 0 {
			X.A[i][i+1] = e.E[i]
		} else if e.E[i] < 0 {
			X.A[i][i-1] = e.E[i]
		}
	}
	return X
}

func (e *EigenvalueDecomposition) Getd() []float64 {
	return e.D
}
