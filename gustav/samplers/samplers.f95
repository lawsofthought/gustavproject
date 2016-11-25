subroutine hdpmm_get_counts(S, R, Sk, latent, observed, group, K_max, V, J, N)

    ! Get the S, R, Sk count matrices for the HDPMM Gibbs sampler.

    implicit none

    integer, intent(in) :: K_max, V, J, N

    integer(kind=8), dimension(N), intent(in) :: latent 
    integer(kind=8), dimension(N), intent(in) :: observed 
    integer(kind=8), dimension(N), intent(in) :: group

    integer(kind=8), dimension(0:K_max-1, 0:V-1), intent(out) :: S
    integer(kind=8), dimension(0:J-1, 0:K_max-1), intent(out) :: R
    integer(kind=8), dimension(0:K_max-1), intent(out) :: Sk

    integer :: i

    S = 0
    R = 0
    Sk = 0

    do i=1,N
        S(latent(i), observed(i)) = S(latent(i), observed(i)) + 1
        R(group(i), latent(i)) = R(group(i), latent(i)) + 1
        Sk(latent(i)) = Sk(latent(i)) + 1
    end do

end subroutine hdpmm_get_counts

subroutine hdpmm_latents_gibbs_sampler(latent, &
                                       observed, & 
                                       group, & 
                                       S, &
                                       R, &
                                       Sk, &
                                       seed,&
                                       psi, & 
				       K_max, &
                                       m, & 
                                       a, & 
                                       b, & 
                                       V, &
                                       J, &
                                       N)


    ! Gibbs sampler for the latent variables in the HDPMM.
    ! This version does *not* take the short cut that is possible when K_rep < K_max.
 
    ! This one generates the permutation index and the random values.

    implicit none

    ! Declare variables
    ! =================
    
    integer(kind=8), intent(in) :: V 
    integer(kind=8), intent(in) :: J
    integer(kind=8), intent(in) :: N

    integer(kind=8) :: K_max

    ! integer arrays of observed, latent, and grouping variables
    integer(kind=8), dimension(0:N-1), intent(inout) :: latent 
    integer(kind=8), dimension(0:N-1), intent(in) :: observed 
    integer(kind=8), dimension(0:N-1), intent(in) :: group

    integer(kind=8), dimension(0:K_max-1, 0:V-1), intent(inout) :: S
    integer(kind=8), dimension(0:J-1, 0:K_max-1), intent(inout) :: R
    integer(kind=8), dimension(0:K_max-1), intent(inout) :: Sk

    integer, intent(in) :: seed

    real(kind=8), dimension(0:N-1) :: rndvals
    integer(kind=8), dimension(0:N-1) :: permutation

    ! hyperparameters of the dirichlet distribtion prior on the (phi) topic distributions
    real(kind=8), dimension(0:V-1), intent(in) :: psi
    real(kind=8), intent(in) :: b

    ! hyperparameters of the dirichlet distribtion prior on the (pi) topic mixtures
    real(kind=8), intent(in) :: a
    real(kind=8), dimension(0:K_max-1), intent(in) :: m

    ! Scratch variables for loop counters, etc.
    integer(kind=8) :: i
    integer(kind=8) :: k
    integer(kind=8) :: j_index
    integer(kind=8) :: v_index
    integer(kind=8) :: current_k
    integer(kind=8) :: sampled_k

    real(kind=8), dimension(0:K_max-1) :: f
    real(kind=8) :: likelihood
    real(kind=8) :: prior

    ! ==================================================================

    call get_latent_rndvals_permutation(permutation,&
                                        rndvals,&
                                        N,&
                                        seed)

    do i = 0,N-1

            j_index = group(permutation(i))
            v_index = observed(permutation(i))
            current_k = latent(permutation(i)) 

            S(current_k, v_index) = S(current_k, v_index) - 1
            R(j_index, current_k) = R(j_index, current_k) - 1
            Sk(current_k) = Sk(current_k) - 1

            do k = 0,K_max-1

                likelihood = (S(k, v_index) + b*psi(v_index)) / (Sk(k) + b)

                prior = R(j_index, k) + a*m(k)

                if (k == 0) then
                    f(k) = likelihood * prior
                else
                    f(k) = f(k-1) + likelihood * prior
                end if

            end do

            call bisection_sampler(sampled_k, f, rndvals(i), K_max-1)
            
            latent(permutation(i)) = sampled_k

            S(sampled_k, v_index) = S(sampled_k, v_index) + 1
            R(j_index, sampled_k) = R(j_index, sampled_k) + 1
            Sk(sampled_k) = Sk(sampled_k) + 1

    end do


end subroutine hdpmm_latents_gibbs_sampler

subroutine hdpmm_latents_gibbs_sampler_alt(latent, &
                                       observed, & 
                                       group, & 
                                       K_rep_new,&
                                       S, &
                                       R, &
                                       Sk, &
                                       seed,&
                                       psi, & 
                                       K_rep, &
				       K_max, &
                                       m, & 
                                       a, & 
                                       b, & 
                                       V, &
                                       J, &
                                       N)

    ! Gibbs sampler for the latent variables in the HDPMM.
    ! This version *does* take the short cut that is possible when K_rep < K_max.

    ! THIS ONE SEEMS TO BE LEADING TO CRASHES.
    ! USE WITH CAUTION, IF AT ALL.
 
    implicit none

    ! Declare variables
    ! =================
    
    integer(kind=8), intent(in) :: V 
    integer(kind=8), intent(in) :: J
    integer(kind=8), intent(in) :: N

    integer(kind=8) :: K_max
    ! The three K_rep variables may be unnecessary ...
    integer(kind=8), intent(in):: K_rep
    integer(kind=8), intent(out):: K_rep_new
    integer(kind=8) :: K_rep_internal

    ! integer arrays of observed, latent, and grouping variables
    integer(kind=8), dimension(0:N-1), intent(inout) :: latent 
    integer(kind=8), dimension(0:N-1), intent(in) :: observed 
    integer(kind=8), dimension(0:N-1), intent(in) :: group

    integer(kind=8), dimension(0:K_max-1, 0:V-1), intent(inout) :: S
    integer(kind=8), dimension(0:J-1, 0:K_max-1), intent(inout) :: R
    integer(kind=8), dimension(0:K_max-1), intent(inout) :: Sk

    integer, intent(in) :: seed

    real(kind=8), dimension(0:N-1) :: rndvals
    integer(kind=8), dimension(0:N-1) :: permutation

    ! hyperparameters of the dirichlet distribtion prior on the (phi) topic distributions
    real(kind=8), dimension(0:V-1), intent(in) :: psi
    real(kind=8), intent(in) :: b

    ! hyperparameters of the dirichlet distribtion prior on the (pi) topic mixtures
    real(kind=8), intent(in) :: a
    real(kind=8), dimension(0:K_max-1), intent(in) :: m

    ! Scratch variables for loop counters, etc.
    integer(kind=8) :: i
    integer(kind=8) :: k
    integer :: k_rep_i
    integer(kind=8) :: j_index
    integer(kind=8) :: v_index
    integer(kind=8) :: current_k
    integer(kind=8) :: sampled_k

    real(kind=8), dimension(0:K_max-1) :: f
    real(kind=8) :: likelihood
    real(kind=8) :: prior
    real(kind=8) :: m_ri

    real(kind=8) :: fu, mu, mu_alt, mu_maxval, mu_minval, fk, threshold, new_rndval

    ! ==================================================================

    call get_latent_rndvals_permutation(permutation,&
                                        rndvals,&
                                        N,&
                                        seed)

    K_rep_internal = K_rep
    do i = 0,N-1

            f = 0.0d+0 ! re-set to zero

            j_index = group(permutation(i))
            v_index = observed(permutation(i))
            current_k = latent(permutation(i)) 

            S(current_k, v_index) = S(current_k, v_index) - 1
            R(j_index, current_k) = R(j_index, current_k) - 1
            Sk(current_k) = Sk(current_k) - 1


            do k = 0,K_rep_internal-1

                likelihood = (S(k, v_index) + b*psi(v_index)) / (Sk(k) + b)

                prior = R(j_index, k) + a*m(k)

                if (k == 0) then
                    f(k) = likelihood * prior
                else
                    f(k) = f(k-1) + likelihood * prior
                end if

            end do

	    fk = f(K_rep_internal-1)
	    mu = sum(m(K_rep_internal:K_max-1))

            threshold = fk/(fk+(psi(v_index) * a * mu))

	    if (rndvals(i) < threshold) then

                    if (K_rep_internal == 1) then
                        sampled_k = 0
                    else
                        new_rndval = rndvals(i)/threshold

			sampled_k = 0
			do
				if (new_rndval <= f(sampled_k)/fk) exit
				sampled_k = sampled_k + 1
			end do
		    end if

	    else
                    new_rndval = (rndvals(i) - threshold)/(1-threshold)

		    m_ri = m(K_rep_internal)
		    do
			if (new_rndval <= m_ri/mu) exit

			if (K_rep_internal == K_max-1) then
				write (*,*) 'K_rep_internal trouble?', K_rep_internal
			end if
			k_rep_internal = k_rep_internal + 1
			m_ri = m_ri + m(K_rep_internal)
		    end do

		    sampled_k = K_rep_internal

		    K_rep_internal = K_rep_internal + 1


	    end if
            
            latent(permutation(i)) = sampled_k

            S(sampled_k, v_index) = S(sampled_k, v_index) + 1
            R(j_index, sampled_k) = R(j_index, sampled_k) + 1
            Sk(sampled_k) = Sk(sampled_k) + 1

    end do

    K_rep_new = K_rep_internal

end subroutine hdpmm_latents_gibbs_sampler_alt
 
subroutine hdpmm_latents_gibbs_sampler_par2(latent_full, &
                                            observed_full, & 
                                            latent, &
                                            observed, &
                                            group, & 
                                            rndvals,&
                                            a, & 
                                            b, & 
                                            psi, & 
                                            m, & 
                                            K_max, &
                                            V, &
                                            J, &
                                            J_full, &
                                            N, &
                                            N_full)

    ! Gibbs sampler for the latent variables in the HDPMM.
    ! This version takes the short cut that is possible when K_rep < K_max.
    ! This is to be used in distributed samplers.

    implicit none
 
    integer(kind=8), intent(in) :: V 
    integer(kind=8), intent(in) :: J, J_full
    integer(kind=8), intent(in) :: N, N_full
    integer(kind=8), intent(in) :: K_max

    integer(kind=8) :: K_rep

    integer(kind=8), dimension(0:N_full-1), intent(in) :: latent_full
    integer(kind=8), dimension(0:N_full-1), intent(in) :: observed_full

    integer(kind=8), dimension(0:N-1), intent(inout) :: latent
    integer(kind=8), dimension(0:N-1), intent(in) :: observed
    integer(kind=8), dimension(0:N-1), intent(in) :: group

    integer(kind=8), dimension(0:K_max-1, 0:V-1) :: S
    integer(kind=8), dimension(0:K_max-1) :: Sk
    integer(kind=8), dimension(0:J-1, 0:K_max-1) :: R

    real(kind=8), dimension(0:N-1), intent(in) :: rndvals

    ! hyperparameters of the dirichlet distribtion prior on the (phi) topic distributions
    real(kind=8), dimension(0:V-1), intent(in) :: psi
    real(kind=8), intent(in) :: b

    ! hyperparameters of the dirichlet distribtion prior on the (pi) topic mixtures
    real(kind=8), intent(in) :: a
    real(kind=8), dimension(0:K_max-1), intent(in) :: m
    
    ! Scratch variables for loop counters, etc.
    integer(kind=8) :: i
    integer(kind=8) :: k
    integer(kind=8) :: j_index
    integer(kind=8) :: v_index
    integer(kind=8) :: current_k
    integer(kind=8) :: sampled_k

    real(kind=8), dimension(0:K_max-1) :: f
    real(kind=8) :: likelihood
    real(kind=8) :: prior
    real(kind=8) :: m_ri

    real(kind=8) :: mu, fk, threshold, new_rndval

    ! Do counts
    ! =========
    S = 0
    Sk = 0

    do i=0,N_full-1
        S(latent_full(i), observed_full(i)) = S(latent_full(i), observed_full(i)) + 1
        Sk(latent_full(i)) = Sk(latent_full(i)) + 1
    end do

    K_rep = maxval(latent_full) + 1

    R = 0
    do i=0,N-1
        R(group(i), latent(i)) = R(group(i), latent(i)) + 1
    end do
    ! ==========

    do i = 0,N-1

            j_index = group(i)
            v_index = observed(i)
            current_k = latent(i) 

            S(current_k, v_index) = S(current_k, v_index) - 1
            R(j_index, current_k) = R(j_index, current_k) - 1
            Sk(current_k) = Sk(current_k) - 1

            do k = 0,K_max-1

                likelihood = (S(k, v_index) + b*psi(v_index)) / (Sk(k) + b)

                prior = R(j_index, k) + a*m(k)

                if (k == 0) then
                    f(k) = likelihood * prior
                else
                    f(k) = f(k-1) + likelihood * prior
                end if

            end do

            call bisection_sampler(sampled_k, f, rndvals(i), K_max-1)
            
            latent(i) = sampled_k

            S(sampled_k, v_index) = S(sampled_k, v_index) + 1
            R(j_index, sampled_k) = R(j_index, sampled_k) + 1
            Sk(sampled_k) = Sk(sampled_k) + 1

    end do

end subroutine hdpmm_latents_gibbs_sampler_par2

! ===============================================================================
! ====================== HYPERPARAMETERS ========================================
! ===============================================================================

subroutine polya_sampler_c2(c_out, sigma_s_colsums, indx, indx_length, stirling_matrix_dim, c, V, seed)

    use random

    implicit none

    ! =============
    integer, intent(in) :: indx_length, stirling_matrix_dim
    integer, dimension(0:indx_length-1), intent(in) :: indx

    integer(kind=8), intent(in) :: V
    integer(kind=8), dimension(0:V-1), intent(in):: sigma_s_colsums
    integer, intent(in) :: seed
    real(kind=8), intent(in) :: c

    ! =============

    real(kind=8), intent(out) :: c_out

    ! =============

    integer(kind=8), dimension(3) :: rnd_seeds_long
    integer, dimension(3) :: rnd_seeds
    real(kind=8) :: tau

    integer(kind=8) :: sigma_q_sum
    real(kind=8) :: shape_parameter, scale_parameter

    ! =========================================================

    call rinteger(rnd_seeds_long, int(3, kind=8), int(101, kind=8), int(10001, kind=8), seed)
    rnd_seeds = int(rnd_seeds_long, kind=4)

    call sigma_sampler_c2(sigma_q_sum, &
		          sigma_s_colsums, &
			  indx, &
			  indx_length,&
			  stirling_matrix_dim,&
			  rnd_seeds(1),&
			  c,&
			  V)
	
    call tau_sampler_c(tau, sum(sigma_s_colsums), rnd_seeds(2), c)

    shape_parameter = sigma_q_sum + 1.0d+0
    scale_parameter = 1/(1.0d+0 - log(tau))
    
    call init_genrand(rnd_seeds(3))
    c_out = rand_gamma(shape_parameter, scale_parameter)

end subroutine polya_sampler_c2


subroutine polya_sampler_c(c_out, sigma_s_colsums, c, V, seed)

    use random

    implicit none

    ! =============
    integer(kind=8), intent(in) :: V
    integer(kind=8), dimension(0:V-1), intent(in):: sigma_s_colsums
    integer, intent(in) :: seed
    real(kind=8), intent(in) :: c

    ! =============

    real(kind=8), intent(out) :: c_out

    ! =============

    integer(kind=8), dimension(3) :: rnd_seeds_long
    integer, dimension(3) :: rnd_seeds
    real(kind=8) :: tau

    integer(kind=8) :: sigma_q_sum
    real(kind=8) :: shape_parameter, scale_parameter

    ! =========================================================

    call rinteger(rnd_seeds_long, int(3, kind=8), int(101, kind=8), int(10001, kind=8), seed)
    rnd_seeds = int(rnd_seeds_long, kind=4)

    call sigma_sampler_c(sigma_q_sum,&
                         maxval(sigma_s_colsums),& 
                         sigma_s_colsums,& 
                         rnd_seeds(1),&
                         c,&
                         V)

    call tau_sampler_c(tau, sum(sigma_s_colsums), rnd_seeds(2), c)

    shape_parameter = sigma_q_sum + 1.0d+0
    scale_parameter = 1/(1.0d+0 - log(tau))
    
    call init_genrand(rnd_seeds(3))
    c_out = rand_gamma(shape_parameter, scale_parameter)

end subroutine polya_sampler_c

subroutine polya_sampler_am(omega, a_out, m_out, R, m, a, J, K, vargamma, seed)

    use random

    implicit none

    ! =============
    integer(kind=8), intent(in) :: J
    integer(kind=8), intent(in) :: K
    integer(kind=8), dimension(0:J-1, 0:K-1), intent(in) :: R
    integer, intent(in) :: seed
    real(kind=8) :: vargamma, c

    ! =============

    real(kind=8), dimension(0:K-1), intent(in) :: m
    real(kind=8), dimension(0:K-1), intent(out) :: m_out
    real(kind=8), intent(in) :: a
    real(kind=8), intent(out) :: a_out

    ! =============

    real(kind=8), dimension(0:K-2), intent(out) :: omega

    ! =============

    integer(kind=8), dimension(4) :: rnd_seeds_long
    integer, dimension(4) :: rnd_seeds
    integer(kind=8), dimension(0:K-1) :: sigma_r_colsums
    real(kind=8), dimension(0:J-1) :: tau

    ! =========================================================

    call rinteger(rnd_seeds_long, int(4, kind=8), int(101, kind=8), int(10001, kind=8), seed)
    rnd_seeds = int(rnd_seeds_long, kind=4)

    call sigma_sampler(sigma_r_colsums,&
                       maxval(R),& 
                       R,& 
                       rnd_seeds(1),&
		       J,&
		       K,&
                       m,&
                       a)

    call gdd_sampler(m_out, omega, sigma_r_colsums, vargamma, rnd_seeds(2), K)
    call tau_sampler(tau, sum(R, 2), a, rnd_seeds(3), J)
    call polya_concentration_sampler(a_out, sum(sigma_r_colsums), tau, J, rnd_seeds(4))

end subroutine polya_sampler_am

subroutine polya_sampler_bpsi(sigma_s_colsums, b_out, psi_out, S, psi, b, K, V, c, seed)

    use random

    implicit none

    ! =============
    integer(kind=8), intent(in) :: K
    integer(kind=8), intent(in) :: V
    integer(kind=8), dimension(0:K-1, 0:V-1), intent(in) :: S
    integer, intent(in) :: seed
    real(kind=8) :: c

    ! =============the

    real(kind=8), dimension(0:V-1), intent(in) :: psi
    real(kind=8), dimension(0:V-1), intent(out) :: psi_out
    real(kind=8), intent(in) :: b
    real(kind=8), intent(out) :: b_out

    ! =============

    integer(kind=8), dimension(4) :: rnd_seeds_long
    integer, dimension(4) :: rnd_seeds
    integer(kind=8), dimension(0:V-1), intent(out) :: sigma_s_colsums
    real(kind=8), dimension(0:K-1) :: tau

    ! =========================================================

    call rinteger(rnd_seeds_long, int(4, kind=8), int(101, kind=8), int(10001, kind=8), seed)
    rnd_seeds = int(rnd_seeds_long, kind=4)

    call sigma_sampler(sigma_s_colsums,&
                       maxval(S),& 
                       S,& 
                       rnd_seeds(1),&
		       K,&
		       V,&
                       psi,&
                       b)
    call ddirichlet_sampler(psi_out, sigma_s_colsums, c/real(V, kind=8), rnd_seeds(2), V)
    call tau_sampler(tau, sum(S, 2), b, rnd_seeds(3), K)
    call polya_concentration_sampler(b_out, sum(sigma_s_colsums), tau, K, rnd_seeds(4))

end subroutine polya_sampler_bpsi

subroutine sigma_sampler(sigma_col_sum, stirling_dim, S, rnd_seed, k, v, psi, b)

    use random
    
    implicit none

    ! Input arguments
    integer(kind=8), intent(in) :: stirling_dim
    integer(kind=8), intent(in) :: K
    integer(kind=8), intent(in) :: V
    integer(kind=8), dimension(0:K-1, 0:V-1), intent(in) :: S
    integer, intent(in) :: rnd_seed
    real(kind=8) :: rnd_val
    real(kind=8), dimension(0:V-1), intent(in) :: psi
    real(kind=8), intent(in) :: b

    ! Internal variables
    real(kind = 8), dimension(0:stirling_dim, 0:stirling_dim) :: stirling_matrix
    integer(kind=8) :: sigma_sample
    integer(kind=8) :: v_index, k_index

    ! Output variables
    integer(kind=8), dimension(0:V-1), intent(out) :: sigma_col_sum

    ! ###########################################################################

    call get_normalized_stirling_numbers(stirling_matrix, stirling_dim)

    call init_genrand(rnd_seed)

    sigma_col_sum = 0

    ! This could be openmped with "parallel do private(k_index, rnd_val, sigma_sample)
    do v_index = 0,V-1
        do k_index = 0,K-1

		if (S(k_index, v_index) /= 0) then

                    rnd_val = rand()

                    call get_sigma_sample(sigma_sample, &
                                          stirling_matrix, &
                                          stirling_dim, &
                                          S(k_index, v_index), &
                                          rnd_val, &
                                          psi(v_index), &
                                          b)


		else
			sigma_sample = 0

		end if
                sigma_col_sum(v_index) = sigma_col_sum(v_index) + sigma_sample

        end do 
    
    end do

end subroutine sigma_sampler

subroutine polya_sampler_bpsi2(sigma_s_colsums, b_out, psi_out, S, indx, indx_length, stirling_matrix_dim, psi, b, K, V, c, seed)

    use random

    implicit none

    ! =============
    integer, intent(in) :: indx_length, stirling_matrix_dim
    integer, dimension(0:indx_length-1), intent(in) :: indx
    integer(kind=8), intent(in) :: K
    integer(kind=8), intent(in) :: V
    integer(kind=8), dimension(0:K-1, 0:V-1), intent(in) :: S
    integer, intent(in) :: seed
    real(kind=8) :: c

    ! =============the

    real(kind=8), dimension(0:V-1), intent(in) :: psi
    real(kind=8), dimension(0:V-1), intent(out) :: psi_out
    real(kind=8), intent(in) :: b
    real(kind=8), intent(out) :: b_out

    ! =============

    integer(kind=8), dimension(4) :: rnd_seeds_long
    integer, dimension(4) :: rnd_seeds
    integer(kind=8), dimension(0:V-1), intent(out) :: sigma_s_colsums
    real(kind=8), dimension(0:K-1) :: tau

    ! =========================================================

    call rinteger(rnd_seeds_long, int(4, kind=8), int(101, kind=8), int(10001, kind=8), seed)
    rnd_seeds = int(rnd_seeds_long, kind=4)

    call sigma_sampler2(sigma_s_colsums,&
                        S, &
                        indx, &
                        indx_length, &
                        stirling_matrix_dim,&
                        rnd_seeds(1),&
		        K,&
		        V,&
                        psi,&
                        b)
               
    call ddirichlet_sampler(psi_out, sigma_s_colsums, c/real(V, kind=8), rnd_seeds(2), V)
    call tau_sampler(tau, sum(S, 2), b, rnd_seeds(3), K)
    call polya_concentration_sampler(b_out, sum(sigma_s_colsums), tau, K, rnd_seeds(4))

end subroutine polya_sampler_bpsi2

subroutine sigma_sampler2(sigma_col_sum, &
                          S, &
                          indx, &
                          indx_length, &
                          stirling_matrix_dim, &
                          rnd_seed, &
                          K, &
                          V, &
                          psi, &
                          b)

    ! This accomplishes the same as sigma_sampler but without calculating the,
    ! possibly huge, stirling_matrix. 
    ! It uses get_normalized_stirling_numbers2 and not get_normalized_stirling_numbers.

    use random
    
    implicit none

    ! Input arguments
    integer, intent(in) :: indx_length, stirling_matrix_dim
    integer, dimension(0:indx_length-1), intent(in) :: indx
    integer(kind=8), intent(in) :: K
    integer(kind=8), intent(in) :: V
    integer(kind=8), dimension(0:K-1, 0:V-1), intent(in) :: S
    integer, intent(in) :: rnd_seed
    real(kind=8), dimension(0:V-1), intent(in) :: psi
    real(kind=8), intent(in) :: b


    ! Internal variables
    real(kind=8) :: rnd_val
    real(kind = 8), dimension(0:indx_length-1, 0:stirling_matrix_dim) :: stirling_matrix

    integer(kind=8) :: sigma_sample
    integer :: v_index, k_index

    ! Output variables
    integer(kind=8), dimension(0:V-1), intent(out) :: sigma_col_sum

    ! ###########################################################################

    call get_normalized_stirling_numbers2(stirling_matrix, &
                                          indx, &
                                          maxval(indx), &
                                          size(indx))

    call init_genrand(rnd_seed)

    sigma_col_sum = 0

    ! This could be openmped with "parallel do private(k_index, rnd_val, sigma_sample)
    do v_index = 0,V-1
        do k_index = 0,K-1

		if (S(k_index, v_index) /= 0) then

                    rnd_val = rand()

                    call get_sigma_sample2(sigma_sample, &
                                           stirling_matrix, &
                                           indx, &
                                           indx_length, &
                                           stirling_matrix_dim, &
                                           S(k_index, v_index), &
                                           rnd_val, &
                                           psi(v_index), &
                                           b)


		else
			sigma_sample = 0

		end if
                sigma_col_sum(v_index) = sigma_col_sum(v_index) + sigma_sample

        end do 
    
    end do

end subroutine sigma_sampler2


subroutine sigma_sampler_c2(sigmasum, &
		            sigma_s, &
			    indx, &
			    indx_length,&
			    stirling_matrix_dim,&
			    rnd_seed,&
			    c,&
			    V)
	
    use random
    
    implicit none

    ! Input arguments
    integer, intent(in) :: indx_length, stirling_matrix_dim
    integer, dimension(0:indx_length-1), intent(in) :: indx

    integer(kind=8), intent(in) :: V
    integer(kind=8), dimension(0:V-1), intent(in) :: sigma_s
    integer, intent(in) :: rnd_seed
    real(kind=8), intent(in) :: c

    ! Internal variables
    real(kind=8) :: rnd_val
    real(kind = 8), dimension(0:indx_length-1, 0:stirling_matrix_dim) :: stirling_matrix
    integer(kind=8) :: sigma_sample
    integer(kind=8) :: v_index
    real(kind=8) :: psi

    ! Output variables
    integer(kind=8), intent(out) :: sigmasum

    ! ###########################################################################

    sigmasum = 0

    psi = 1/real(V, kind=8)

    call get_normalized_stirling_numbers2(stirling_matrix, &
	                                  indx,&
					  maxval(indx),&
					  size(indx))

    call init_genrand(rnd_seed)

    do v_index = 0,V-1

	if (sigma_s(v_index) /= 0) then

	    rnd_val = rand()

	    call get_sigma_sample2(sigma_sample, &
				   stirling_matrix, &
				   indx,&
				   indx_length,&
				   stirling_matrix_dim, &
				   sigma_s(v_index), &
				   rnd_val, &
				   psi, &
				   c)

	else
		sigma_sample = 0

	end if

	sigmasum = sigmasum + sigma_sample
    
    end do

end subroutine sigma_sampler_c2

subroutine sigma_sampler_c(sigmasum, stirling_dim, sigma_s, rnd_seed, c, V)

    use random
    
    implicit none

    ! Input arguments
    integer(kind=8), intent(in) :: stirling_dim
    integer(kind=8), intent(in) :: V
    integer(kind=8), dimension(0:V-1), intent(in) :: sigma_s
    integer, intent(in) :: rnd_seed
    real(kind=8), intent(in) :: c

    ! Internal variables
    real(kind=8) :: rnd_val
    real(kind = 8), dimension(0:stirling_dim, 0:stirling_dim) :: stirling_matrix
    integer(kind=8) :: sigma_sample
    integer(kind=8) :: v_index
    real(kind=8) :: psi

    ! Output variables
    integer(kind=8), intent(out) :: sigmasum

    ! ###########################################################################

    sigmasum = 0

    psi = 1/real(V, kind=8)

    call get_normalized_stirling_numbers(stirling_matrix, stirling_dim)

    call init_genrand(rnd_seed)

    do v_index = 0,V-1

	if (sigma_s(v_index) /= 0) then

	    rnd_val = rand()

	    call get_sigma_sample(sigma_sample, &
				  stirling_matrix, &
				  stirling_dim, &
				  sigma_s(v_index), &
				  rnd_val, &
				  psi, &
				  c)

	else
		sigma_sample = 0

	end if

	sigmasum = sigmasum + sigma_sample
    
    end do

end subroutine sigma_sampler_c

subroutine tau_sampler_c(tau, sigma_s_sum, rnd_seed, c)

    use random
    
    implicit none

    ! Input arguments
    integer(kind=8), intent(in) :: sigma_s_sum
    integer, intent(in) :: rnd_seed
    real(kind=8), intent(in) :: c

    ! Output variables
    real(kind=8), intent(out) :: tau

    ! ###########################################################################

    call init_genrand(rnd_seed)
    tau = rand_beta(c, real(sigma_s_sum, kind=8))
    
end subroutine tau_sampler_c

subroutine get_sigma_sample2(sampled_k, &
                             stirling_matrix, &
                             indx, &
                             indx_length, &
                             stirling_matrix_dim, &
                             k, &
                             rndval, &
                             psi_v, &
                             b)

    use random

    implicit none

    integer, intent(in) :: indx_length, stirling_matrix_dim
    integer, dimension(0:indx_length-1), intent(in) :: indx
    real(kind = 8), dimension(0:indx_length-1, 0:stirling_matrix_dim), intent(in) :: stirling_matrix
 
    integer(kind=8), intent(in) :: k
    real(kind=8), dimension(0:k) :: q

    real(kind=8), intent(in) :: psi_v
    real(kind=8), intent(in) :: b
    real(kind=8), intent(in) :: rndval

    integer(kind=8), intent(out) :: sampled_k
    integer(kind=8) :: i

    integer :: ind

    real(kind=8) :: f

    ! ===============================
    
    ! k is the value of the S(row, col) array
    ! That will be in indx somewhere. Find it.
    ind = 0
    do
        if (indx(ind) == k) exit
        ind = ind + 1
    end do

    ! Ok. indx(ind) == k. So we want to use stirling_matrix(ind,:) below.

    q = 0.0
    do i = 0, k

       f = stirling_matrix(ind, i) * (b*psi_v)**i
       if (i == 0) then
               q(i) = f
       else 
               q(i) = q(i-1) + f
       end if

    end do

    call bisection_sampler(sampled_k, q, rndval, k)
    
end subroutine get_sigma_sample2

subroutine get_sigma_sample(sampled_k, stirling_matrix, max_dim, k, rndval, psi_v, b)

    use random

    implicit none

    integer(kind=8), intent(in) :: max_dim
    integer(kind=8), intent(in) :: k
    real(kind=8), dimension(0:k) :: q

    real(kind=8), intent(in) :: psi_v
    real(kind=8), intent(in) :: b
    real(kind=8), intent(in) :: rndval

    integer(kind=8), intent(out) :: sampled_k
    integer(kind=8) :: i

    real(kind = 8), dimension(0:max_dim, 0:max_dim), intent(in) :: stirling_matrix

    real(kind=8) :: f

    ! ===============================

    q = 0.0
    do i = 0, k

       f = stirling_matrix(k, i) * (b*psi_v)**i
       if (i == 0) then
               q(i) = f
       else 
               q(i) = q(i-1) + f
       end if

    end do

    call bisection_sampler(sampled_k, q, rndval, k)
    
end subroutine get_sigma_sample

subroutine gdd_sampler(m, samples, sigmasum, var_gamma, seed, V)

    ! Draw samples from a Generalized Dirichlet distribution.

    use random
    implicit none

    integer(kind=8), intent(in) :: V
    integer(kind=8), dimension(0:V-1), intent(in) :: sigmasum

    real(kind=8), dimension(V-1) :: alpha
    real(kind=8), dimension(V-1) :: beta
    real(kind=8), dimension(V-1), intent(out) :: samples

    real(kind=8), dimension(0:V-1), intent(out) :: m

    integer, intent(in) :: seed
    real(kind=8), intent(in) :: var_gamma
    real(kind=8) :: stick

    integer(kind=8) :: v_i, i

    ! =========================================================

    do v_i = 0, V-2
        alpha(v_i+1) = 1 + sigmasum(v_i)
        beta(v_i+1) = var_gamma + sum(sigmasum(v_i+1:V-1))
    end do

    call get_rbeta(samples, alpha, beta, V-1, seed)

    stick = 1.0d+0
    do i = 0, V-2
        m(i) = stick * samples(i+1)
        stick = (1-samples(i+1)) * stick
    end do
    m(V-1) = stick

end subroutine 

subroutine gamma_sampler(vargamma, omega, K, seed, prior_shape_parameter, prior_scale_parameter)

        use random
        implicit none

        integer(kind=8), intent(in) :: K
        real(kind=8), intent(in), dimension(K) :: omega

        integer, intent(in) :: seed

        real(kind=8), intent(in) :: prior_shape_parameter
        real(kind=8), intent(in) :: prior_scale_parameter

        real(kind=8) :: shape_parameter
        real(kind=8) :: scale_parameter

        ! #################
        real(kind=8), intent(out) :: vargamma

        ! #################
        call init_genrand(seed)

        shape_parameter = prior_shape_parameter + real(K, kind=8)
        scale_parameter = 1/(1/prior_scale_parameter - sum(log(1-omega)))

        vargamma = rand_gamma(shape_parameter, scale_parameter)

end subroutine gamma_sampler

subroutine tau_sampler(tau, Rj, a, seed, J)

    use random
    implicit none

    integer(kind=8), intent(in) :: J
    integer(kind=8), dimension(0:J-1), intent(in) :: Rj
    real(kind=8), intent(in) :: a

    real(kind=8), dimension(0:J-1), intent(out) :: tau
    real(kind=8), dimension(0:J-1) :: a_row

    integer, intent(in) :: seed

    integer(kind=8) :: v_i, i

    ! =========================================================

    a_row = 0.01 + a ! All a_row equal to a (plus eps)

    call get_rbeta(tau, a_row, 0.01 + real(Rj, kind=8), J, seed)

end subroutine 

subroutine polya_concentration_sampler(b, sigmasum, tau, K, seed)
 
        use random
        implicit none

        integer, intent(in) :: seed
        integer(kind=8), intent(in) :: K
        integer(kind=8), intent(in) :: sigmasum
        real(kind=8), dimension(K)  :: tau
        real(kind=8), intent(out) :: b

        real(kind=8) :: shape_parameter, scale_parameter

        shape_parameter = sigmasum + 1.0d+0
        scale_parameter = 1/(1.0d+0 - sum(log(tau)))

        call init_genrand(seed)
        b = rand_gamma(shape_parameter, scale_parameter)

end subroutine polya_concentration_sampler

subroutine ddirichlet_sampler(psi, sigmarows, var_gamma, seed, V)

    use random
    implicit none

    integer(kind=8), intent(in) :: V
    integer(kind=8), dimension(0:V-1), intent(in) :: sigmarows

    real(kind=8), dimension(0:V-1), intent(out) :: psi

    integer, intent(in) :: seed
    real(kind=8), intent(in) :: var_gamma

    ! =========================================================

    call get_rdirichlet(psi, sigmarows + var_gamma, seed, V)

end subroutine 

! ===============================================================================
! ====================== FOR TESTING ONLY =======================================
! ===============================================================================

subroutine hdpmm_latents_gibbs_sampler2(latent, &
                                       observed, & 
                                       group, & 
                                       S, &
                                       R, &
                                       Sk, &
				       rndvals, &
				       permutation, &
                                       psi, & 
				       K_max, &
                                       m, & 
                                       a, & 
                                       b, & 
                                       V, &
                                       J, &
				       index_N, &
                                       N)


    ! THIS IS JUST USED FOR TESTING THE DISTRIBUTED SAMPLER (i.e. hdpmm_latents_gibbs_sampler_par2)
    ! =============================================================================================

    ! Gibbs sampler for the latent variables in the HDPMM.
    ! This version does *not* take the short cut that is possible when K_rep < K_max.

    ! This one drops in the permutation index and the random values.
 
    implicit none

    ! Declare variables
    ! =================
    
    integer(kind=8), intent(in) :: V 
    integer(kind=8), intent(in) :: J
    integer(kind=8), intent(in) :: N
    integer(kind=8), intent(in) :: index_N

    integer(kind=8) :: K_max

    ! integer arrays of observed, latent, and grouping variables
    integer(kind=8), dimension(0:N-1), intent(inout) :: latent 
    integer(kind=8), dimension(0:N-1), intent(in) :: observed 
    integer(kind=8), dimension(0:N-1), intent(in) :: group

    integer(kind=8), dimension(0:K_max-1, 0:V-1), intent(inout) :: S
    integer(kind=8), dimension(0:J-1, 0:K_max-1), intent(inout) :: R
    integer(kind=8), dimension(0:K_max-1), intent(inout) :: Sk

    real(kind=8), dimension(0:index_N-1), intent(in):: rndvals
    integer(kind=8), dimension(0:index_N-1), intent(in):: permutation

    ! hyperparameters of the dirichlet distribtion prior on the (phi) topic distributions
    real(kind=8), dimension(0:V-1), intent(in) :: psi
    real(kind=8), intent(in) :: b

    ! hyperparameters of the dirichlet distribtion prior on the (pi) topic mixtures
    real(kind=8), intent(in) :: a
    real(kind=8), dimension(0:K_max-1), intent(in) :: m

    ! Scratch variables for loop counters, etc.
    integer(kind=8) :: i
    integer(kind=8) :: k
    integer(kind=8) :: j_index
    integer(kind=8) :: v_index
    integer(kind=8) :: current_k
    integer(kind=8) :: sampled_k

    real(kind=8), dimension(0:K_max-1) :: f
    real(kind=8) :: likelihood
    real(kind=8) :: prior

    ! ==================================================================

    do i = 0,index_N-1

            j_index = group(permutation(i))
            v_index = observed(permutation(i))
            current_k = latent(permutation(i)) 

            S(current_k, v_index) = S(current_k, v_index) - 1
            R(j_index, current_k) = R(j_index, current_k) - 1
            Sk(current_k) = Sk(current_k) - 1

            do k = 0,K_max-1

                likelihood = (S(k, v_index) + b*psi(v_index)) / (Sk(k) + b)

                prior = R(j_index, k) + a*m(k)

                if (k == 0) then
                    f(k) = likelihood * prior
                else
                    f(k) = f(k-1) + likelihood * prior
                end if

            end do

            call bisection_sampler(sampled_k, f, rndvals(i), K_max-1)
            
            latent(permutation(i)) = sampled_k

            S(sampled_k, v_index) = S(sampled_k, v_index) + 1
            R(j_index, sampled_k) = R(j_index, sampled_k) + 1
            Sk(sampled_k) = Sk(sampled_k) + 1

    end do

end subroutine hdpmm_latents_gibbs_sampler2


