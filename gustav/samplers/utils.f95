subroutine bisection_sampler(sampled_k, q, rndval, k)

    ! zero index bisection sampler

    implicit none

    integer(kind=8), intent(in) :: k
    real(kind=8), dimension(0:k), intent(in) :: q

    real(kind=8), intent(in) :: rndval
    integer(kind=8), intent(out) :: sampled_k

    real(kind=8) :: qk, qn
    integer(kind=8) :: i, j, midpoint

    qn = q(k)

    i = 0
    j = k
    do 
        midpoint = (i+j)/2   

        qk = q(midpoint)

        if (rndval <= qk/qn) then 
            j = midpoint 
        else
            i = midpoint
        end if

        if (i+1 >= j) exit

    end do

    if (rndval <= q(i)/qn) then
        sampled_k = i
    else
        sampled_k = j
    end if

end subroutine bisection_sampler

subroutine get_stirling_numbers(stirling_matrix, k)

  ! return a (k+1 x k+1) matrix of unsigned stirling numbers of the first kind.

  implicit none

  integer(kind = 8), intent(in) :: k

  integer(kind = 8) :: i
  integer(kind = 8) :: j
  integer(kind = 8), dimension(0:k, 0:k), intent(out) :: stirling_matrix

  ! initialization
  stirling_matrix = 0 ! initialize the whole matrix to 0
  stirling_matrix(0, 0) = 1
  forall (i=1:k) 
      stirling_matrix(i, i) = 1
      stirling_matrix(i, 0) = 0
  end forall

  do i = 2,k
    do j = 1, i-1
      stirling_matrix(i, j) = stirling_matrix(i-1, j-1) + (i-1)*stirling_matrix(i-1, j)
    end do
  end do
 
end subroutine get_stirling_numbers

subroutine get_normalized_stirling_numbers(normalized_stirling_matrix, k)

  ! return a (k+1 x k+1) matrix of unsigned stirling numbers of the first kind.
  ! rows are normalized by dividing my their maximum value
  ! for some problems, the relative values per rows are all that are needed.  
  ! this is good because the number tend to get huge

  implicit none

  integer(kind = 8), intent(in) :: k

  integer(kind = 8) :: i
  integer(kind = 8) :: j
  real(kind = 8), dimension(0:k, 0:k), intent(out) :: normalized_stirling_matrix

  ! initialization
  normalized_stirling_matrix = 0.0 ! initialize the whole matrix to 0
  normalized_stirling_matrix(0, 0) = 1
  forall (i=1:k) 
      normalized_stirling_matrix(i, i) = 1
      normalized_stirling_matrix(i, 0) = 0
  end forall

  do i = 1,k
    do j = 1, k
      normalized_stirling_matrix(i, j) = normalized_stirling_matrix(i-1, j-1) + (i-1)*normalized_stirling_matrix(i-1, j)
    end do
    normalized_stirling_matrix(i, :) = normalized_stirling_matrix(i, :)/maxval(normalized_stirling_matrix(i, :))
  end do
 
end subroutine get_normalized_stirling_numbers

subroutine get_normalized_stirling_numbers2(normalized_stirling_matrix, indx, N, K)

  ! Return a (k x N+1) matrix of unsigned stirling numbers of the first kind.
  ! Rows are normalized by dividing my their maximum value.
  ! For some problems, the relative values per rows are all that are needed.  
  ! This is good because the number tend to get huge
  ! Here, we return the rows of the N+1 x N+1 matrix that correspond to the
  ! indices in indx only.
  ! This allows us to avoid calculating a N+1 x N+1 matrix when only some rows 
  ! are ultimately needed.

  ! The indx array must be sorted and a unique array of integers.
  ! The integer N must equal max(indx).
  ! The integer K is the length of indx.

  implicit none

  integer, intent(in) :: K
  integer, intent(in) :: N
  integer, dimension(0:K-1), intent(in) :: indx
  real(kind = 8), dimension(0:K-1, 0:N), intent(out) :: normalized_stirling_matrix

  real(kind = 8), dimension(0:N) :: u, u_new
  integer :: i, ind

  ! initialization
  normalized_stirling_matrix = 0.0 ! initialize the whole matrix to 0

  ind = 0
  i = 0

  u = 0.0 
  u(0) = 1.0

  if (indx(ind) == 0) then
      normalized_stirling_matrix(ind,0:N) = u
      ind = ind + 1
  end if

  do i = 1,N
      u_new = u * (i-1)
    u_new(1:N) = u_new(1:N) + u(0:N-1)
    u = u_new/maxval(u_new)

    if (indx(ind) == i) then
      normalized_stirling_matrix(ind,0:N) = u
      ind = ind + 1
    end if

  end do

end subroutine get_normalized_stirling_numbers2


subroutine get_rbeta(rndvals, a, b, k, seed)

    ! Return values drawn from k beta distributions.
    ! Specifically, for each of the k values in the arrays a and b, 
    ! do 
    !    rndvals(k_i) = rand_beta(a(k_i), b(k_i))
    ! and then return rndvals.
    
    use random
    implicit none

    integer, intent(in) :: seed
        integer(kind = 8), intent(in) :: k

        real(kind=8), dimension(k), intent(in) :: a
        real(kind=8), dimension(k), intent(in) :: b

    real(kind=8), dimension(k), intent(out) :: rndvals

    integer(kind=8) :: k_i

    call init_genrand(seed)
    do k_i=1,k
        rndvals(k_i) = rand_beta(a(k_i), b(k_i))
    end do

end subroutine get_rbeta


subroutine get_random_integers(rndvals, n, min_int, max_int, seed)
    
    use random
    implicit none

    integer, intent(in) :: seed
    integer(kind=8), intent(in) :: min_int, max_int
    integer(kind=8), intent(in) :: n
    integer(kind=8), dimension(n), intent(out) :: rndvals

    call rinteger(rndvals, n, min_int, max_int, seed)

end subroutine get_random_integers

subroutine get_rgamma(rndval, shape_parameter, scale_parameter, seed)

    use random
    implicit none

    real(kind=8), intent(out) :: rndval
    real(kind=8), dimension(1) :: rndvals
    real(kind=8), intent(in) :: shape_parameter, scale_parameter
    integer, intent(in)  :: seed

    integer(kind=8), parameter :: one = 1

        call rgamma(rndvals, one, shape_parameter, scale_parameter, seed)
        rndval = rndvals(1)

end subroutine get_rgamma

subroutine get_rdirichlet(rndval, dirichlet_parameters, seed, k)

    use random
    implicit none

    integer(kind=8), intent(in) :: k
    real(kind=8), intent(in), dimension(k) :: dirichlet_parameters
    integer, intent(in)  :: seed
    integer(kind=8), parameter :: one = 1

    real(kind=8), dimension(one, k) :: rndvals
    real(kind=8), intent(out), dimension(k) :: rndval

    call rdirichlet(rndvals, one, dirichlet_parameters, k, seed)
    rndval = rndvals(1,1:k)

end subroutine get_rdirichlet 

subroutine sigma_sampler_rnd_vals(rnd_vals, S, rnd_seed, k, v)

    ! useful only as a utils for testing sigma_sampler
    ! Return array of random values

    use random
    
    implicit none

    ! Input arguments
    integer(kind=8), intent(in) :: K
    integer(kind=8), intent(in) :: V
    integer(kind=8), dimension(0:K-1, 0:V-1), intent(in) :: S
    real(kind=8), dimension(0:K-1, 0:V-1), intent(out) :: rnd_vals
    integer, intent(in) :: rnd_seed

    ! Internal variables
    integer(kind=8) :: v_index, k_index

    ! ###########################################################################
    rnd_vals = 0.0d+0

    call init_genrand(rnd_seed)

    do v_index = 0,V-1
        do k_index = 0,K-1

        if (S(k_index, v_index) /= 0) then
                    rnd_vals(k_index, v_index) = rand()
        else
                    rnd_vals(k_index, v_index) = 0.0d+0
        end if

        end do 
    
    end do

end subroutine sigma_sampler_rnd_vals

subroutine sigma_sampler_c_rnd_vals(rnd_vals, sigma_s, rnd_seed, V)

    use random
    
    implicit none

    ! Input arguments
    integer(kind=8), intent(in) :: V
    integer(kind=8), dimension(0:V-1), intent(in) :: sigma_s
    integer, intent(in) :: rnd_seed

    ! Internal variables
    integer(kind=8) :: v_index

    ! Output variables
    real(kind=8), dimension(0:V-1), intent(out) :: rnd_vals

    ! ###########################################################################

    call init_genrand(rnd_seed)

    do v_index = 0,V-1

    if (sigma_s(v_index) /= 0) then
            rnd_vals(v_index) = rand()
    else
            rnd_vals(v_index) = 0.0d+0
    end if
    
    end do

end subroutine sigma_sampler_c_rnd_vals

subroutine get_newseeds(rnd_seeds, N, seed)

    ! Make N new random seeds from seed.

    use random
    implicit none

    integer(kind=8) :: N
    integer :: seed
    integer(kind=8), dimension(N) :: rnd_seeds_long
    integer, dimension(N) :: rnd_seeds

    call rinteger(rnd_seeds_long, N, int(101, kind=8), int(100001, kind=8), seed)
    rnd_seeds = int(rnd_seeds_long, kind=4)

end subroutine get_newseeds

subroutine get_randperm(permutation, N, rnd_seed)

    use random
    implicit none

    integer, intent(in) :: rnd_seed
    integer(kind=8), intent(in) :: N
    integer(kind=8), dimension(N), intent(out) :: permutation

    call rperm(permutation, N, rnd_seed)

    permutation = permutation - 1 ! Make it zero indexed

end subroutine get_randperm

subroutine get_rndvals(rndvals, n, seed)

    use random
    implicit none

    integer(kind=8), intent(in) :: n
    real(kind=8), intent(out), dimension(n) :: rndvals
    integer, intent(in)  :: seed

        call runif(rndvals, n, seed)

end subroutine get_rndvals

subroutine get_latent_rndvals_permutation(permutation, rndvals, N, seed)

    integer(kind=8), intent(in) :: N

    integer, intent(in) :: seed
    integer, dimension(2) :: new_seeds

    real(kind=8), intent(out), dimension(0:N-1) :: rndvals
    integer(kind=8), intent(out), dimension(0:N-1) :: permutation
 
    call get_newseeds(new_seeds, int(2, kind=8), seed)
    call get_randperm(permutation, N, new_seeds(1))
    call get_rndvals(rndvals, N, new_seeds(2))

end subroutine get_latent_rndvals_permutation

!subroutine get_omp_nthreads(nthreads)
!
!    implicit none
!
!    integer, intent(out) :: nthreads
!    integer :: thread_id, omp_get_num_threads, omp_get_thread_num
!
!    !$omp parallel private(thread_id)
!    thread_id = omp_get_thread_num()
!
!    if (thread_id == 0) then
!      nthreads = omp_get_num_threads()
!    end if
!    !$omp end parallel
!
!end subroutine get_omp_nthreads

subroutine get_counts(S, R, Sk, latent, observed, group, V, J, K_max, N)

    implicit none

    integer(kind=8), intent(in) :: V 
    integer(kind=8), intent(in) :: J
    integer(kind=8), intent(in) :: N
    integer(kind=8), intent(in) :: K_max

    integer(kind=8), dimension(0:N-1), intent(in) :: latent 
    integer(kind=8), dimension(0:N-1), intent(in) :: observed 
    integer(kind=8), dimension(0:N-1), intent(in) :: group

    integer(kind=8), dimension(0:K_max-1, 0:V-1), intent(out) :: S
    integer(kind=8), dimension(0:J-1, 0:K_max-1), intent(out) :: R
    integer(kind=8), dimension(0:K_max-1), intent(out) :: Sk

    ! Scratch variables for loop counters, etc.
    integer(kind=8) :: i
    integer(kind=8) :: j_index
    integer(kind=8) :: v_index
    integer(kind=8) :: k_index

    S = 0
    Sk = 0
    R = 0
    do i = 0,N-1

            j_index = group(i)
            v_index = observed(i)
            k_index = latent(i) 

            S(k_index, v_index) = S(k_index, v_index) + 1
            R(j_index, k_index) = R(j_index, k_index) + 1
            Sk(k_index) = Sk(k_index) + 1

    end do


end subroutine get_counts
