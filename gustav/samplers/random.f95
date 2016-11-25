MODULE random

    ! Module for random number generation.
    ! Code taken from different sources, and possibly modified.

    ! ###################################################################################
    ! ###################################################################################

    ! rand, init_genrand are from the following:

        ! A Fortran-program for MT19937: Real number version

        ! Code converted using TO_F90 by Alan Miller
        ! Date: 1999-11-26  Time: 17:09:23
        ! Latest revision - 5 February 2002
        ! A new seed initialization routine has been added based upon the new
        ! C version dated 26 January 2002.
        ! This version assumes that integer overflows do NOT cause crashes.
        ! This version is compatible with Lahey's ELF90 compiler,
        ! and should be compatible with most full Fortran 90 or 95 compilers.
        ! Notice the strange way in which umask is specified for ELF90.

        !   genrand() generates one pseudorandom real number (double) which is
        ! uniformly distributed on [0,1]-interval, for each call.
        ! sgenrand(seed) set initial values to the working area of 624 words.
        ! Before genrand(), sgenrand(seed) must be called once.  (seed is any 32-bit
        ! integer except for 0).
        ! Integer generator is obtained by modifying two lines.
        !   Coded by Takuji Nishimura, considering the suggestions by
        ! Topher Cooper and Marc Rieffel in July-Aug. 1997.

        ! This library is free software; you can redistribute it and/or modify it
        ! under the terms of the GNU Library General Public License as published by
        ! the Free Software Foundation; either version 2 of the License, or (at your
        ! option) any later version.   This library is distributed in the hope that
        ! it will be useful, but WITHOUT ANY WARRANTY; without even the implied
        ! warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
        ! See the GNU Library General Public License for more details.
        ! You should have received a copy of the GNU Library General Public License
        ! along with this library; if not, write to the Free Foundation, Inc.,
        ! 59 Temple Place, Suite 330, Boston, MA 02111-1307  USA

        ! Copyright (C) 1997 Makoto Matsumoto and Takuji Nishimura.
        ! When you use this, send an email to: matumoto@math.keio.ac.jp
        ! with an appropriate reference to your work.

        !***********************************************************************
        ! Fortran translation by Hiroshi Takano.  Jan. 13, 1999.

        !   genrand()      -> double precision function rand()
        !   sgenrand(seed) -> subroutine sgrnd(seed)
        !                     integer seed

        ! This program uses the following standard intrinsics.
        !   ishft(i,n): If n > 0, shifts bits in i by n positions to left.
        !               If n < 0, shifts bits in i by n positions to right.
        !   iand (i,j): Performs logical AND on corresponding bits of i and j.
        !   ior  (i,j): Performs inclusive OR on corresponding bits of i and j.
        !   ieor (i,j): Performs exclusive OR on corresponding bits of i and j.

        !***********************************************************************

    ! ###################################################################################
    ! ###################################################################################


    IMPLICIT NONE

    INTEGER, PARAMETER :: dp = SELECTED_REAL_KIND(12, 60)

    ! Period parameters
    INTEGER, PARAMETER :: n = 624, n1 = n+1, m = 397, mata = -1727483681
    !                                    constant vector a
    INTEGER, PARAMETER :: umask = -2147483647 - 1
    !                                    most significant w-r bits
    INTEGER, PARAMETER :: lmask =  2147483647
    !                                    least significant r bits

    ! Tempering parameters
    INTEGER, PARAMETER :: tmaskb= -1658038656, tmaskc= -272236544

    ! The array for the state vector.
    INTEGER, SAVE      :: mt(0:n-1), mti = n1
    ! mti==N+1 means mt[N] is not initialized

    double precision, parameter :: PI=3.141592653589793238462

    PRIVATE
    PUBLIC :: dp, rand, init_genrand, rand_normal, rand_gamma, &
        rand_beta, rand_dirichlet, rand_integer, runif, rnorm, rgamma, &
        rdirichlet, rinteger, rperm, rand_stickbreaking


CONTAINS

    !***********************************************************************
    SUBROUTINE init_genrand(seed)
        ! This initialization is based upon the multiplier given on p.106 of the
        ! 3rd edition of Knuth, The Art of Computer Programming Vol. 2.

        ! This version assumes that integer overflow does NOT cause a crash.

        INTEGER, INTENT(IN)  :: seed

        INTEGER  :: latest

        mt(0) = seed
        latest = seed
        DO mti = 1, n-1
        latest = IEOR( latest, ISHFT( latest, -30 ) )
        latest = latest * 1812433253 + mti
        mt(mti) = latest
        END DO

        RETURN
    END SUBROUTINE init_genrand

    !***********************************************************************

    FUNCTION rand() RESULT(fn_val)

        implicit none

        REAL (dp) :: fn_val

        INTEGER, SAVE :: mag01(0:1) = (/ 0, mata /)
        !                        mag01(x) = x * MATA for x=0,1
        INTEGER       :: kk, y

        IF(mti >= n) THEN
            !                       generate N words at one time
            IF(mti == n+1) THEN
                !                            if sgrnd() has not been called,
                CALL init_genrand(4357)
                !                              a default initial seed is used
            END IF

            DO  kk = 0, n-m-1
            y = IOR(IAND(mt(kk),umask), IAND(mt(kk+1),lmask))
            mt(kk) = IEOR(IEOR(mt(kk+m), ISHFT(y,-1)),mag01(IAND(y,1)))
            END DO

            DO  kk = n-m, n-2
            y = IOR(IAND(mt(kk),umask), IAND(mt(kk+1),lmask))
            mt(kk) = IEOR(IEOR(mt(kk+(m-n)), ISHFT(y,-1)),mag01(IAND(y,1)))
            END DO

            y = IOR(IAND(mt(n-1),umask), IAND(mt(0),lmask))
            mt(n-1) = IEOR(IEOR(mt(m-1), ISHFT(y,-1)),mag01(IAND(y,1)))
            mti = 0
        END IF

        y = mt(mti)
        mti = mti + 1
        y = IEOR(y, tshftu(y))
        y = IEOR(y, IAND(tshfts(y),tmaskb))
        y = IEOR(y, IAND(tshftt(y),tmaskc))
        y = IEOR(y, tshftl(y))

        IF(y < 0) THEN
            fn_val = (DBLE(y) + 2.0D0**32) / (2.0D0**32 - 1.0D0)
        ELSE
            fn_val = DBLE(y) / (2.0D0**32 - 1.0D0)
        END IF

        RETURN

    END FUNCTION rand


    FUNCTION tshftu(y) RESULT(fn_val)
        INTEGER, INTENT(IN) :: y
        INTEGER             :: fn_val

        fn_val = ISHFT(y,-11)
        RETURN
    END FUNCTION tshftu


    FUNCTION tshfts(y) RESULT(fn_val)
        INTEGER, INTENT(IN) :: y
        INTEGER             :: fn_val

        fn_val = ISHFT(y,7)
        RETURN
    END FUNCTION tshfts


    FUNCTION tshftt(y) RESULT(fn_val)
        INTEGER, INTENT(IN) :: y
        INTEGER             :: fn_val

        fn_val = ISHFT(y,15)
        RETURN
    END FUNCTION tshftt


    FUNCTION tshftl(y) RESULT(fn_val)
        INTEGER, INTENT(IN) :: y
        INTEGER             :: fn_val

        fn_val = ISHFT(y,-18)
        RETURN
    END FUNCTION tshftl

    function rand_normal(mean,stdev) result(c)

        implicit none

        ! Random sample from normal (Gaussian) distribution

        real(kind=8) :: mean
        real(kind=8) :: stdev
        real(kind=8) :: r, c, temp_1, temp_2, theta


        if (stdev <= 0.0d0) then

            Write(*,*) "Standard Deviation must be +ve"

        else

            temp_1 = rand()

            temp_2 = rand()

            r=(-2.0d0*log(temp_1))**0.5

            theta = 2.0d0*PI*temp_2

            c= mean+stdev*r*sin(theta)

        end if     

    end function

    recursive function rand_gamma(shape, scale) result(ans)
        implicit none
        
        !  Return a random sample from a gamma distribution
        
        real(kind=8) :: shape,scale,u,w,d,c,x,xsq,g
        real(kind=8) :: v
        real(kind=8) :: ans 

        if (shape <= 0.0d0) then
            write(*,*) "Shape parameter must be positive", shape
        end if
        if (scale <= 0.0d0) then
            write(*,*) "Scale parameter must be positive", scale
        end if
        
        !    ## Implementation based on "A Simple Method for Generating Gamma Variables"
        !    ## by George Marsaglia and Wai Wan Tsang.  
        !    ## ACM Transactions on Mathematical Software
        !    ## Vol 26, No 3, September 2000, pages 363-372.
        
        if (shape >= 1.0d0) then
            d = shape - 1.0d0/3.0d0
            c = 1.0d0/(9.0d0*d)**0.5
            do while (.true.)
            x = rand_normal(0.0d0, 1.0d0)
            v = 1.0 + c*x
            do while (v <= 0.0d0) 
            x = rand_normal(0.0d0, 1.0d0)
            v = 1.0d0 + c*x
            end do
            v = v*v*v

            u = rand()

            xsq = x*x
            if ((u < 1.0d0 -.0331d0*xsq*xsq) .or. (log(u) < 0.5d0*xsq + d*(1.0d0 - v + log(v))) )then
                ans=scale*d*v
                return 
            end if
            end do
        else
            g = rand_gamma(shape+1.0d0, 1.0d0)
            w = rand()
            ans=scale*g*(w)**(1.0d0/shape)
            return 
        end if
    end function


    function rand_beta(a, b) result(ans)
        implicit none

        ! ## return a random sample from a beta distribution

        double precision a,b,ans,u,v

        if ((a <= 0.0d0) .or. (b <= 0.0d0)) then
            write(*,*) "Beta parameters must be positive", a, b
        end if

        !    ## There are more efficient methods for generating beta samples.
        !    ## However such methods are a little more efficient and much more complicated.
        !    ## For an explanation of why the following method works, see
        !    ## http://www.johndcook.com/distribution_chart.html#gamma_beta

        u = rand_gamma(a, 1.0d0)
        v = rand_gamma(b, 1.0d0)
        ans = u / (u + v)
    end function


    function rand_dirichlet(params, k) result(p)

        implicit none

           real(kind=8), parameter :: scale_param = 1.0

           integer(kind=8) :: k, k_i
           real(kind=8), dimension(k) :: params
           real(kind=8), dimension(k) :: f
           real(kind=8), dimension(k) :: p
           real(kind=8) :: z
       
           z = 0.0
           do k_i=1,k
           f(k_i) = rand_gamma(params(k_i), scale_param)
           z = z + f(k_i)
           end do

           p = f / z

    end function rand_dirichlet


        function rand_integer(min_int, max_int) result(ans)

        implicit none

            ! A single random integer drawn uniformly from the 
            ! range of integers min_int to max_int.
        ! If min_int > max_int, then these two values are swapped.

            integer(kind=8) :: min_int, max_int
            integer(kind=8) :: new_min_int, new_max_int
            real(kind=8) :: u
            integer(kind=8) :: ans

        new_min_int = min(min_int, max_int)
        new_max_int = max(min_int, max_int)
            
            u = rand()

            ans = new_min_int + floor((new_max_int+1-new_min_int)*u)

        end function 

    function rand_stickbreaking(param, K) result(m)

        ! Sample from a stick breaking distribution
        ! where the stick breaking distribution
        ! is a *standard* distribution, i.e. a = 1.0, b = param

        implicit none

        ! in
        real(kind=8) :: param
        integer(kind=8) :: K 

        ! out
        real(kind=8), dimension(K+1) :: m

        ! scratch
        real(kind=8) :: stick 
        integer(kind=8) :: k_i

        ! #############################################

        stick = 1.0d+0
        do k_i=1,K
            m(k_i) = stick * rand_beta(1.0d+0, param)
            stick = stick - m(k_i)
        end do
        m(K+1) = stick 

    end function rand_stickbreaking

        subroutine rand_permutation(order, n)

        !     Generate a random ordering of the integers 1 ... n.
        !     borrowed from: Alan Miller
        !             CSIRO Division of Mathematical & Information Sciences
        !     e-mail: amiller @ bigpond.net.au


        integer(kind=8), intent(in)  :: n
        integer(kind=8), intent(out), dimension(n) :: order

        ! Local variables

        INTEGER :: i, j, k
        real(kind=8)  :: wk

        DO i = 1, n
          order(i) = i
        END DO

        !     Starting at the end, swap the current last indicator with one
        !     randomly chosen from those preceeding it.

        DO i = n, 2, -1
          wk = rand()
          j = 1 + i * wk
          IF (j < i) THEN
            k = order(i)
            order(i) = order(j)
            order(j) = k
          END IF
        END DO

        RETURN
        end subroutine rand_permutation

    ! ############################################################
    ! Return multiple random samples
    ! ############################################################

    subroutine runif(x, N, rnd_seed)

        implicit none

        integer(kind=8), intent(in) :: N
        integer, intent(in) :: rnd_seed
        real(kind=8), dimension(N), intent(out) :: x

        integer(kind=8) :: i

        call init_genrand(rnd_seed)

        do i=1,N
            x(i) = rand()
        end do

    end subroutine


    subroutine rnorm(x, N, mean, sd, rnd_seed)

        implicit none

        integer(kind=8) :: i, N
        real(kind=8), dimension(N), intent(out) :: x

        real(kind=8), intent(in) :: mean
        real(kind=8), intent(in) :: sd
        integer, intent(in) :: rnd_seed

        call init_genrand(rnd_seed)

        do i=1,N
            x(i) = rand_normal(mean, sd)
        end do

    end subroutine

    subroutine rgamma(x, N, shape, scale, rnd_seed)

        implicit none

        integer(kind=8) :: i, N
        real(kind=8), dimension(N), intent(out) :: x

        real(kind=8), intent(in) :: shape
        real(kind=8), intent(in) :: scale

        integer, intent(in) :: rnd_seed

        call init_genrand(rnd_seed)

        do i=1,N
            x(i) = rand_gamma(shape, scale)
        end do

    end subroutine

    subroutine rdirichlet(p, N, params, K, rnd_seed)

        implicit none

        integer(kind=8), intent(in) :: N
        integer(kind=8), intent(in) :: K
        real(kind=8), dimension(K), intent(in) :: params

        integer, intent(in) :: rnd_seed

        real(kind=8), dimension(N, K), intent(out) :: p

        integer(kind=8) :: i

        call init_genrand(rnd_seed)

        do i=1,N
            p(i, 1:K) = rand_dirichlet(params, K)
        end do

    end subroutine


        subroutine rinteger(rndvals, N, min_int, max_int, rnd_seed)

        implicit none

            integer, intent(in) :: rnd_seed
            integer(kind=8), intent(in) :: min_int, max_int
            integer(kind=8), intent(in) :: N
            integer(kind=8), dimension(N), intent(out) :: rndvals

        integer(kind=8) :: i

        call init_genrand(rnd_seed)

            do i=1,N
                rndvals(i) = rand_integer(min_int, max_int)
            end do

        end subroutine rinteger

        subroutine rperm(permutation, N, rnd_seed)

        implicit none

            integer, intent(in) :: rnd_seed
            integer(kind=8), intent(in) :: N
            integer(kind=8), dimension(N), intent(out) :: permutation

        integer(kind=8) :: i

        call init_genrand(rnd_seed)
            call rand_permutation(permutation, N)

        end subroutine rperm

END MODULE random
