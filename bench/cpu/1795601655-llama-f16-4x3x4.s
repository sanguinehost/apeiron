_ZN12_GLOBAL__N_18tinyBLASILi8EDv8_fS1_fffE4gemmILi4ELi3ELi4EEEvlll: # @_ZN12_GLOBAL__N_18tinyBLASILi8EDv8_fS1_fffE4gemmILi4ELi3ELi4EEEvlll
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$328, %rsp                      # imm = 0x148
	.cfi_def_cfa_offset 384
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rdi, 8(%rsp)                   # 8-byte Spill
	testb	$15, %sil
	jne	.LBB1_56
# %bb.1:
	leaq	2(%rdx), %rax
	movabsq	$6148914691236517206, %rcx      # imm = 0x5555555555555556
	movq	%rdx, %r8
	movq	%rsi, %r15
	movl	$1, %esi
	imulq	%rcx
	movq	%rdx, %r13
	shrq	$63, %r13
	addq	%rdx, %r13
	cmpq	$70, %r8
	jl	.LBB1_3
# %bb.2:
	leaq	12(%r13), %rdx
	movabsq	$-6148914691236517205, %rax     # imm = 0xAAAAAAAAAAAAAAAB
	mulxq	%rax, %rsi, %rsi
	shrq	$4, %rsi
.LBB1_3:
	movq	%r13, %rax
	cqto
	idivq	%rsi
	movq	8(%rsp), %r9                    # 8-byte Reload
	cmpq	$1, %rdx
	movq	%rsi, %rdx
	movq	%rax, %rdi
	sbbq	$-1, %rdi
	movq	(%r9), %rax
	movq	%rdi, %rcx
	imulq	%rsi, %rcx
	movq	%rdi, 168(%rsp)                 # 8-byte Spill
	subq	%r13, %rcx
	subq	%rcx, %rdx
	cmpl	$0, (%rax)
	movq	%rdx, 160(%rsp)                 # 8-byte Spill
	je	.LBB1_5
# %bb.4:
	movq	8(%rsp), %r14                   # 8-byte Reload
	movq	%r8, %rbx
	jmp	.LBB1_7
.LBB1_5:
	imulq	%rdi, %rdx
	decq	%rdi
	movq	%rsi, %r12
	imulq	%rdi, %rcx
	addq	%rdx, %rcx
	cmpq	%r13, %rcx
	jne	.LBB1_57
# %bb.6:
	movq	24(%rax), %rdi
	movl	4(%rax), %esi
	movq	%r8, %rbx
	callq	ggml_threadpool_chunk_set@PLT
	movq	8(%rsp), %r14                   # 8-byte Reload
	movq	%r12, %rsi
	movq	(%r14), %rax
.LBB1_7:
	movq	24(%rax), %rdi
	sarq	$4, %r15
	imulq	%r15, %rsi
	movq	%r15, 248(%rsp)                 # 8-byte Spill
	movq	%rsi, 176(%rsp)                 # 8-byte Spill
	callq	ggml_barrier@PLT
	movq	(%r14), %rcx
	movslq	(%rcx), %rax
	cmpq	%rax, 176(%rsp)                 # 8-byte Folded Reload
	jle	.LBB1_55
# %bb.8:
	movq	168(%rsp), %rdi                 # 8-byte Reload
	movq	160(%rsp), %r8                  # 8-byte Reload
	leaq	(%r13,%r13,2), %rcx
	movq	%rbx, %rsi
	addq	%r13, %rsi
	subq	%rcx, %rsi
	leaq	(%rsi,%rsi,2), %r9
	movq	%r8, %rcx
	imulq	%rdi, %rcx
	leaq	-1(%rdi), %rdx
	movq	%rcx, 152(%rsp)                 # 8-byte Spill
	movq	%rdx, 144(%rsp)                 # 8-byte Spill
	movq	8(%rsp), %rbx                   # 8-byte Reload
	movq	%rsi, 240(%rsp)                 # 8-byte Spill
	movq	%r9, 232(%rsp)                  # 8-byte Spill
	.p2align	4, 0x90
.LBB1_9:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_23 Depth 2
                                        #       Child Loop BB1_25 Depth 3
                                        #         Child Loop BB1_34 Depth 4
                                        #         Child Loop BB1_38 Depth 4
                                        #       Child Loop BB1_29 Depth 3
                                        #         Child Loop BB1_44 Depth 4
                                        #         Child Loop BB1_47 Depth 4
	cqto
	idivq	248(%rsp)                       # 8-byte Folded Reload
	movq	%rax, %rcx
	subq	%r8, %rcx
	jge	.LBB1_11
# %bb.10:                               #   in Loop: Header=BB1_9 Depth=1
	movq	%rax, %rcx
	imulq	%rdi, %rcx
	incq	%rax
	movq	%rax, %r10
	subq	%r8, %r10
	jl	.LBB1_13
.LBB1_14:                               #   in Loop: Header=BB1_9 Depth=1
	imulq	144(%rsp), %r10                 # 8-byte Folded Reload
	addq	152(%rsp), %r10                 # 8-byte Folded Reload
	movq	%r10, %rax
	movq	%rcx, %rdi
	subq	%rsi, %rdi
	jl	.LBB1_16
.LBB1_17:                               #   in Loop: Header=BB1_9 Depth=1
	leaq	(%r9,%rdi,2), %rdi
	movq	%rdi, 40(%rsp)                  # 8-byte Spill
	movq	%rax, %rdi
	subq	%rsi, %rdi
	jl	.LBB1_19
.LBB1_20:                               #   in Loop: Header=BB1_9 Depth=1
	leaq	(%r9,%rdi,2), %rax
	jmp	.LBB1_21
	.p2align	4, 0x90
.LBB1_11:                               #   in Loop: Header=BB1_9 Depth=1
	imulq	144(%rsp), %rcx                 # 8-byte Folded Reload
	addq	152(%rsp), %rcx                 # 8-byte Folded Reload
	incq	%rax
	movq	%rax, %r10
	subq	%r8, %r10
	jge	.LBB1_14
.LBB1_13:                               #   in Loop: Header=BB1_9 Depth=1
	imulq	%rdi, %rax
	movq	%rcx, %rdi
	subq	%rsi, %rdi
	jge	.LBB1_17
.LBB1_16:                               #   in Loop: Header=BB1_9 Depth=1
	leaq	(%rcx,%rcx,2), %rdi
	movq	%rdi, 40(%rsp)                  # 8-byte Spill
	movq	%rax, %rdi
	subq	%rsi, %rdi
	jge	.LBB1_20
.LBB1_19:                               #   in Loop: Header=BB1_9 Depth=1
	leaq	(%rax,%rax,2), %rax
.LBB1_21:                               #   in Loop: Header=BB1_9 Depth=1
	movq	%rax, 16(%rsp)                  # 8-byte Spill
	movq	%rdx, %rax
	shlq	$4, %rax
	movq	32(%rbx), %r10
	movq	%rax, 192(%rsp)                 # 8-byte Spill
	movq	16(%rsp), %rax                  # 8-byte Reload
	leaq	-1(%r10), %rdi
	cmpq	%r9, %rax
	cmovlq	%rax, %r9
	movq	24(%rbx), %rax
	shrq	$3, %rdi
	incq	%rdi
	movl	%edi, %r8d
	andl	$7, %r8d
	movq	%rax, 184(%rsp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rdi, 320(%rsp)                 # 8-byte Spill
	andq	$-8, %rdi
	cmpq	%rsi, %rcx
	movq	%rsi, %rax
	cmovlq	%rcx, %rax
	movq	%rdi, 200(%rsp)                 # 8-byte Spill
	shlq	$6, %rdx
	leaq	(%rax,%rcx,2), %rax
	leaq	8(,%rax,4), %rcx
	leaq	4(,%rax,4), %rax
	movq	%rcx, 272(%rsp)                 # 8-byte Spill
	movq	56(%rbx), %rcx
	movq	%rcx, 224(%rsp)                 # 8-byte Spill
	movq	%rax, 264(%rsp)                 # 8-byte Spill
	movq	40(%rsp), %rax                  # 8-byte Reload
	leaq	(,%rax,4), %rax
	movq	%rax, 256(%rsp)                 # 8-byte Spill
	leaq	12(%rdx), %rax
	movq	%rax, 96(%rsp)                  # 8-byte Spill
	leaq	8(%rdx), %rax
	movq	%rax, 88(%rsp)                  # 8-byte Spill
	leaq	4(%rdx), %rax
	movq	%rdx, 104(%rsp)                 # 8-byte Spill
	movq	%rax, 80(%rsp)                  # 8-byte Spill
	movq	%r8, 312(%rsp)                  # 8-byte Spill
                                        # kill: def $r8d killed $r8d killed $r8 def $r8
	shll	$5, %r8d
	xorl	%eax, %eax
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	movq	%r9, 216(%rsp)                  # 8-byte Spill
	movq	%r10, 72(%rsp)                  # 8-byte Spill
	movq	%r8, 208(%rsp)                  # 8-byte Spill
	.p2align	4, 0x90
.LBB1_23:                               #   Parent Loop BB1_9 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB1_25 Depth 3
                                        #         Child Loop BB1_34 Depth 4
                                        #         Child Loop BB1_38 Depth 4
                                        #       Child Loop BB1_29 Depth 3
                                        #         Child Loop BB1_44 Depth 4
                                        #         Child Loop BB1_47 Depth 4
	movq	40(%rsp), %rax                  # 8-byte Reload
	movq	%rax, %r12
	cmpq	%r9, %rax
	jge	.LBB1_27
# %bb.24:                               #   in Loop: Header=BB1_23 Depth=2
	movq	8(%rsp), %rdx                   # 8-byte Reload
	movq	272(%rsp), %r14                 # 8-byte Reload
	movq	264(%rsp), %rbp                 # 8-byte Reload
	movq	40(%rdx), %rdi
	movq	8(%rdx), %rax
	movq	48(%rdx), %r15
	movq	16(%rdx), %rcx
	movq	%rdi, %r11
	imulq	96(%rsp), %r11                  # 8-byte Folded Reload
	movq	%rdi, %r12
	movq	%rdi, %rbx
	imulq	%r15, %r14
	leaq	(,%r15,4), %rsi
	imulq	%r15, %rbp
	leaq	(%rsi,%rsi,2), %r9
	leaq	224(%rcx,%r14), %rdx
	addq	%rcx, %r14
	leaq	224(%rax,%r11), %r8
	addq	%rax, %r11
	movq	%r8, 120(%rsp)                  # 8-byte Spill
	imulq	88(%rsp), %r12                  # 8-byte Folded Reload
	imulq	80(%rsp), %rbx                  # 8-byte Folded Reload
	imulq	104(%rsp), %rdi                 # 8-byte Folded Reload
	movq	%r11, 304(%rsp)                 # 8-byte Spill
	leaq	224(%rax,%r12), %r10
	leaq	224(%rax,%rbx), %r13
	leaq	224(%rax,%rdi), %r8
	addq	%rax, %r12
	addq	%rax, %rbx
	addq	%rax, %rdi
	movq	%r12, 296(%rsp)                 # 8-byte Spill
	movq	%rbx, 288(%rsp)                 # 8-byte Spill
	movq	%rdi, 112(%rsp)                 # 8-byte Spill
	leaq	224(%rcx,%rbp), %rdi
	addq	%rcx, %rbp
	imulq	256(%rsp), %r15                 # 8-byte Folded Reload
	movq	184(%rsp), %r11                 # 8-byte Reload
	leaq	224(%rcx,%r15), %rax
	addq	%rcx, %r15
	movq	48(%rsp), %rcx                  # 8-byte Reload
	orq	192(%rsp), %rcx                 # 8-byte Folded Reload
	movq	%r9, 128(%rsp)                  # 8-byte Spill
	movq	40(%rsp), %r12                  # 8-byte Reload
	leaq	(%r11,%rcx,4), %rbx
	movq	%r15, %r11
	movq	%rbx, 280(%rsp)                 # 8-byte Spill
	jmp	.LBB1_25
	.p2align	4, 0x90
.LBB1_26:                               #   in Loop: Header=BB1_25 Depth=3
	movq	(%rsp), %r11                    # 8-byte Reload
	movq	%rbx, %r15
	leaq	1(%r12), %rbx
	leaq	2(%r12), %r9
	vxorps	%xmm0, %xmm0, %xmm0
	vxorps	%xmm2, %xmm2, %xmm2
	vxorps	%xmm6, %xmm6, %xmm6
	vxorps	%xmm3, %xmm3, %xmm3
	vxorps	%xmm7, %xmm7, %xmm7
	vxorps	%xmm8, %xmm8, %xmm8
	vxorps	%xmm10, %xmm10, %xmm10
	vxorps	%xmm9, %xmm9, %xmm9
	vxorps	%xmm11, %xmm11, %xmm11
	vxorps	%xmm1, %xmm1, %xmm1
	vxorps	%xmm5, %xmm5, %xmm5
	vxorps	%xmm4, %xmm4, %xmm4
.LBB1_40:                               #   in Loop: Header=BB1_25 Depth=3
	vextractf128	$1, %ymm11, %xmm12
	vextractf128	$1, %ymm9, %xmm13
	movq	224(%rsp), %rsi                 # 8-byte Reload
	vaddps	%xmm11, %xmm12, %xmm11
	vaddps	%xmm9, %xmm13, %xmm9
	movq	%rsi, %rcx
	imulq	%r12, %rcx
	imulq	%rsi, %rbx
	imulq	%rsi, %r9
	addq	$3, %r12
	vshufpd	$1, %xmm11, %xmm11, %xmm12      # xmm12 = xmm11[1,0]
	vshufpd	$1, %xmm9, %xmm9, %xmm13        # xmm13 = xmm9[1,0]
	vaddps	%xmm12, %xmm11, %xmm11
	vextractf128	$1, %ymm10, %xmm12
	vaddps	%xmm13, %xmm9, %xmm9
	vinsertps	$76, %xmm11, %xmm9, %xmm13 # xmm13 = xmm11[1],xmm9[1],zero,zero
	vunpcklps	%xmm9, %xmm11, %xmm9    # xmm9 = xmm11[0],xmm9[0],xmm11[1],xmm9[1]
	vaddps	%xmm10, %xmm12, %xmm10
	vshufpd	$1, %xmm10, %xmm10, %xmm12      # xmm12 = xmm10[1,0]
	vaddps	%xmm12, %xmm10, %xmm10
	vextractf128	$1, %ymm8, %xmm12
	vaddps	%xmm8, %xmm12, %xmm8
	vshufpd	$1, %xmm8, %xmm8, %xmm12        # xmm12 = xmm8[1,0]
	vaddps	%xmm12, %xmm8, %xmm8
	vshufps	$85, %xmm10, %xmm8, %xmm12      # xmm12 = xmm8[1,1],xmm10[1,1]
	vmovlhps	%xmm10, %xmm8, %xmm8            # xmm8 = xmm8[0],xmm10[0]
	vshufps	$36, %xmm8, %xmm9, %xmm8        # xmm8 = xmm9[0,1],xmm8[2,0]
	vshufps	$36, %xmm12, %xmm13, %xmm12     # xmm12 = xmm13[0,1],xmm12[2,0]
	vextractf128	$1, %ymm3, %xmm9
	vaddps	%xmm12, %xmm8, %xmm8
	vmovups	%xmm8, (%r15,%rcx,4)
	vextractf128	$1, %ymm7, %xmm8
	vaddps	%xmm3, %xmm9, %xmm3
	movq	128(%rsp), %rcx                 # 8-byte Reload
	vshufpd	$1, %xmm3, %xmm3, %xmm9         # xmm9 = xmm3[1,0]
	vaddps	%xmm7, %xmm8, %xmm7
	vaddps	%xmm3, %xmm9, %xmm3
	addq	%rcx, %rdx
	addq	%rcx, %rdi
	addq	%rcx, %rax
	addq	%rcx, %r14
	addq	%rcx, %rbp
	addq	%rcx, %r11
	movq	216(%rsp), %rcx                 # 8-byte Reload
	vshufpd	$1, %xmm7, %xmm7, %xmm8         # xmm8 = xmm7[1,0]
	vaddps	%xmm7, %xmm8, %xmm7
	vextractf128	$1, %ymm6, %xmm8
	vinsertps	$76, %xmm7, %xmm3, %xmm9 # xmm9 = xmm7[1],xmm3[1],zero,zero
	vunpcklps	%xmm3, %xmm7, %xmm3     # xmm3 = xmm7[0],xmm3[0],xmm7[1],xmm3[1]
	vaddps	%xmm6, %xmm8, %xmm6
	vshufpd	$1, %xmm6, %xmm6, %xmm8         # xmm8 = xmm6[1,0]
	vaddps	%xmm6, %xmm8, %xmm6
	vextractf128	$1, %ymm2, %xmm8
	vaddps	%xmm2, %xmm8, %xmm2
	vshufpd	$1, %xmm2, %xmm2, %xmm8         # xmm8 = xmm2[1,0]
	vaddps	%xmm2, %xmm8, %xmm2
	vshufps	$85, %xmm6, %xmm2, %xmm8        # xmm8 = xmm2[1,1],xmm6[1,1]
	vmovlhps	%xmm6, %xmm2, %xmm2             # xmm2 = xmm2[0],xmm6[0]
	vshufps	$36, %xmm8, %xmm9, %xmm8        # xmm8 = xmm9[0,1],xmm8[2,0]
	vshufps	$36, %xmm2, %xmm3, %xmm2        # xmm2 = xmm3[0,1],xmm2[2,0]
	vaddps	%xmm2, %xmm8, %xmm2
	vmovups	%xmm2, (%r15,%rbx,4)
	vextractf128	$1, %ymm0, %xmm2
	movq	%r15, %rbx
	vaddps	%xmm0, %xmm2, %xmm0
	vshufpd	$1, %xmm0, %xmm0, %xmm2         # xmm2 = xmm0[1,0]
	vaddps	%xmm2, %xmm0, %xmm0
	vextractf128	$1, %ymm5, %xmm2
	vaddps	%xmm5, %xmm2, %xmm2
	vextractf128	$1, %ymm1, %xmm5
	vshufpd	$1, %xmm2, %xmm2, %xmm3         # xmm3 = xmm2[1,0]
	vaddps	%xmm3, %xmm2, %xmm2
	vextractf128	$1, %ymm4, %xmm3
	vaddps	%xmm1, %xmm5, %xmm1
	vshufpd	$1, %xmm1, %xmm1, %xmm5         # xmm5 = xmm1[1,0]
	vaddps	%xmm4, %xmm3, %xmm3
	vaddps	%xmm5, %xmm1, %xmm1
	vshufpd	$1, %xmm3, %xmm3, %xmm4         # xmm4 = xmm3[1,0]
	vinsertps	$76, %xmm0, %xmm1, %xmm5 # xmm5 = xmm0[1],xmm1[1],zero,zero
	vunpcklps	%xmm1, %xmm0, %xmm0     # xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
	vaddps	%xmm4, %xmm3, %xmm3
	vshufps	$85, %xmm2, %xmm3, %xmm4        # xmm4 = xmm3[1,1],xmm2[1,1]
	vmovlhps	%xmm2, %xmm3, %xmm2             # xmm2 = xmm3[0],xmm2[0]
	vshufps	$36, %xmm4, %xmm5, %xmm4        # xmm4 = xmm5[0,1],xmm4[2,0]
	vshufps	$36, %xmm2, %xmm0, %xmm0        # xmm0 = xmm0[0,1],xmm2[2,0]
	vaddps	%xmm4, %xmm0, %xmm0
	vmovups	%xmm0, (%r15,%r9,4)
	cmpq	%rcx, %r12
	jge	.LBB1_27
.LBB1_25:                               #   Parent Loop BB1_9 Depth=1
                                        #     Parent Loop BB1_23 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB1_34 Depth 4
                                        #         Child Loop BB1_38 Depth 4
	movq	%r11, (%rsp)                    # 8-byte Spill
	cmpq	$0, 72(%rsp)                    # 8-byte Folded Reload
	jle	.LBB1_26
# %bb.31:                               #   in Loop: Header=BB1_25 Depth=3
	movq	%r14, 32(%rsp)                  # 8-byte Spill
	movq	%rbp, 24(%rsp)                  # 8-byte Spill
	vxorps	%xmm0, %xmm0, %xmm0
	cmpq	$57, 72(%rsp)                   # 8-byte Folded Reload
	movq	120(%rsp), %rcx                 # 8-byte Reload
	jae	.LBB1_33
# %bb.32:                               #   in Loop: Header=BB1_25 Depth=3
	vxorps	%xmm2, %xmm2, %xmm2
	vxorps	%xmm6, %xmm6, %xmm6
	vxorps	%xmm3, %xmm3, %xmm3
	vxorps	%xmm7, %xmm7, %xmm7
	vxorps	%xmm8, %xmm8, %xmm8
	vxorps	%xmm10, %xmm10, %xmm10
	vxorps	%xmm9, %xmm9, %xmm9
	vxorps	%xmm11, %xmm11, %xmm11
	vxorps	%xmm1, %xmm1, %xmm1
	vxorps	%xmm5, %xmm5, %xmm5
	vxorps	%xmm4, %xmm4, %xmm4
	xorl	%ebp, %ebp
	jmp	.LBB1_35
	.p2align	4, 0x90
.LBB1_33:                               #   in Loop: Header=BB1_25 Depth=3
	movq	200(%rsp), %rsi                 # 8-byte Reload
	vxorps	%xmm2, %xmm2, %xmm2
	vxorps	%xmm6, %xmm6, %xmm6
	vxorps	%xmm3, %xmm3, %xmm3
	vxorps	%xmm7, %xmm7, %xmm7
	vxorps	%xmm8, %xmm8, %xmm8
	vxorps	%xmm10, %xmm10, %xmm10
	vxorps	%xmm9, %xmm9, %xmm9
	vxorps	%xmm11, %xmm11, %xmm11
	vxorps	%xmm1, %xmm1, %xmm1
	vxorps	%xmm5, %xmm5, %xmm5
	vxorps	%xmm4, %xmm4, %xmm4
	xorl	%ebp, %ebp
	.p2align	4, 0x90
.LBB1_34:                               #   Parent Loop BB1_9 Depth=1
                                        #     Parent Loop BB1_23 Depth=2
                                        #       Parent Loop BB1_25 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	vmovups	-224(%rax,%rbp,4), %ymm12
	vmovups	-224(%r8,%rbp,4), %ymm15
	vmovups	-224(%rdi,%rbp,4), %ymm13
	vmovups	-224(%rdx,%rbp,4), %ymm14
	vfmadd231ps	%ymm12, %ymm15, %ymm11  # ymm11 = (ymm15 * ymm12) + ymm11
	vfmadd231ps	%ymm13, %ymm15, %ymm7   # ymm7 = (ymm15 * ymm13) + ymm7
	vfmadd231ps	%ymm15, %ymm14, %ymm0   # ymm0 = (ymm14 * ymm15) + ymm0
	vmovups	-224(%r13,%rbp,4), %ymm15
	vfmadd231ps	%ymm12, %ymm15, %ymm9   # ymm9 = (ymm15 * ymm12) + ymm9
	vfmadd231ps	%ymm13, %ymm15, %ymm3   # ymm3 = (ymm15 * ymm13) + ymm3
	vfmadd231ps	%ymm15, %ymm14, %ymm1   # ymm1 = (ymm14 * ymm15) + ymm1
	vmovups	-224(%r10,%rbp,4), %ymm15
	vfmadd231ps	%ymm12, %ymm15, %ymm10  # ymm10 = (ymm15 * ymm12) + ymm10
	vfmadd231ps	%ymm13, %ymm15, %ymm6   # ymm6 = (ymm15 * ymm13) + ymm6
	vfmadd231ps	%ymm15, %ymm14, %ymm5   # ymm5 = (ymm14 * ymm15) + ymm5
	vmovups	-224(%rcx,%rbp,4), %ymm15
	vfmadd231ps	%ymm12, %ymm15, %ymm8   # ymm8 = (ymm15 * ymm12) + ymm8
	vfmadd231ps	%ymm13, %ymm15, %ymm2   # ymm2 = (ymm15 * ymm13) + ymm2
	vfmadd231ps	%ymm14, %ymm15, %ymm4   # ymm4 = (ymm15 * ymm14) + ymm4
	vmovups	-192(%rax,%rbp,4), %ymm12
	vmovups	-192(%r8,%rbp,4), %ymm13
	vmovups	-192(%rdi,%rbp,4), %ymm14
	vmovups	-192(%rdx,%rbp,4), %ymm15
	vfmadd231ps	%ymm12, %ymm13, %ymm11  # ymm11 = (ymm13 * ymm12) + ymm11
	vfmadd231ps	%ymm14, %ymm13, %ymm7   # ymm7 = (ymm13 * ymm14) + ymm7
	vfmadd231ps	%ymm13, %ymm15, %ymm0   # ymm0 = (ymm15 * ymm13) + ymm0
	vmovups	-192(%r13,%rbp,4), %ymm13
	vfmadd231ps	%ymm12, %ymm13, %ymm9   # ymm9 = (ymm13 * ymm12) + ymm9
	vfmadd231ps	%ymm14, %ymm13, %ymm3   # ymm3 = (ymm13 * ymm14) + ymm3
	vfmadd231ps	%ymm13, %ymm15, %ymm1   # ymm1 = (ymm15 * ymm13) + ymm1
	vmovups	-192(%r10,%rbp,4), %ymm13
	vfmadd231ps	%ymm12, %ymm13, %ymm10  # ymm10 = (ymm13 * ymm12) + ymm10
	vfmadd231ps	%ymm14, %ymm13, %ymm6   # ymm6 = (ymm13 * ymm14) + ymm6
	vfmadd231ps	%ymm13, %ymm15, %ymm5   # ymm5 = (ymm15 * ymm13) + ymm5
	vmovups	-192(%rcx,%rbp,4), %ymm13
	vfmadd231ps	%ymm12, %ymm13, %ymm8   # ymm8 = (ymm13 * ymm12) + ymm8
	vfmadd231ps	%ymm14, %ymm13, %ymm2   # ymm2 = (ymm13 * ymm14) + ymm2
	vfmadd231ps	%ymm15, %ymm13, %ymm4   # ymm4 = (ymm13 * ymm15) + ymm4
	vmovups	-160(%rax,%rbp,4), %ymm12
	vmovups	-160(%r8,%rbp,4), %ymm14
	vmovups	-160(%rdi,%rbp,4), %ymm13
	vmovups	-160(%rdx,%rbp,4), %ymm15
	vfmadd231ps	%ymm12, %ymm14, %ymm11  # ymm11 = (ymm14 * ymm12) + ymm11
	vfmadd231ps	%ymm13, %ymm14, %ymm7   # ymm7 = (ymm14 * ymm13) + ymm7
	vfmadd231ps	%ymm14, %ymm15, %ymm0   # ymm0 = (ymm15 * ymm14) + ymm0
	vmovups	-160(%r13,%rbp,4), %ymm14
	vfmadd231ps	%ymm12, %ymm14, %ymm9   # ymm9 = (ymm14 * ymm12) + ymm9
	vfmadd231ps	%ymm13, %ymm14, %ymm3   # ymm3 = (ymm14 * ymm13) + ymm3
	vfmadd231ps	%ymm14, %ymm15, %ymm1   # ymm1 = (ymm15 * ymm14) + ymm1
	vmovups	-160(%r10,%rbp,4), %ymm14
	vfmadd231ps	%ymm12, %ymm14, %ymm10  # ymm10 = (ymm14 * ymm12) + ymm10
	vfmadd231ps	%ymm13, %ymm14, %ymm6   # ymm6 = (ymm14 * ymm13) + ymm6
	vfmadd231ps	%ymm14, %ymm15, %ymm5   # ymm5 = (ymm15 * ymm14) + ymm5
	vmovups	-160(%rcx,%rbp,4), %ymm14
	vfmadd231ps	%ymm12, %ymm14, %ymm8   # ymm8 = (ymm14 * ymm12) + ymm8
	vfmadd231ps	%ymm13, %ymm14, %ymm2   # ymm2 = (ymm14 * ymm13) + ymm2
	vfmadd231ps	%ymm15, %ymm14, %ymm4   # ymm4 = (ymm14 * ymm15) + ymm4
	vmovups	-128(%rax,%rbp,4), %ymm12
	vmovups	-128(%r8,%rbp,4), %ymm13
	vmovups	-128(%rdi,%rbp,4), %ymm14
	vmovups	-128(%rdx,%rbp,4), %ymm15
	vfmadd231ps	%ymm12, %ymm13, %ymm11  # ymm11 = (ymm13 * ymm12) + ymm11
	vfmadd231ps	%ymm14, %ymm13, %ymm7   # ymm7 = (ymm13 * ymm14) + ymm7
	vfmadd231ps	%ymm13, %ymm15, %ymm0   # ymm0 = (ymm15 * ymm13) + ymm0
	vmovups	-128(%r13,%rbp,4), %ymm13
	vfmadd231ps	%ymm12, %ymm13, %ymm9   # ymm9 = (ymm13 * ymm12) + ymm9
	vfmadd231ps	%ymm14, %ymm13, %ymm3   # ymm3 = (ymm13 * ymm14) + ymm3
	vfmadd231ps	%ymm13, %ymm15, %ymm1   # ymm1 = (ymm15 * ymm13) + ymm1
	vmovups	-128(%r10,%rbp,4), %ymm13
	vfmadd231ps	%ymm12, %ymm13, %ymm10  # ymm10 = (ymm13 * ymm12) + ymm10
	vfmadd231ps	%ymm14, %ymm13, %ymm6   # ymm6 = (ymm13 * ymm14) + ymm6
	vfmadd231ps	%ymm13, %ymm15, %ymm5   # ymm5 = (ymm15 * ymm13) + ymm5
	vmovups	-128(%rcx,%rbp,4), %ymm13
	vfmadd231ps	%ymm12, %ymm13, %ymm8   # ymm8 = (ymm13 * ymm12) + ymm8
	vfmadd231ps	%ymm14, %ymm13, %ymm2   # ymm2 = (ymm13 * ymm14) + ymm2
	vfmadd231ps	%ymm15, %ymm13, %ymm4   # ymm4 = (ymm13 * ymm15) + ymm4
	vmovups	-96(%rax,%rbp,4), %ymm12
	vmovups	-96(%r8,%rbp,4), %ymm15
	vmovups	-96(%rdi,%rbp,4), %ymm13
	vmovups	-96(%rdx,%rbp,4), %ymm14
	vfmadd231ps	%ymm12, %ymm15, %ymm11  # ymm11 = (ymm15 * ymm12) + ymm11
	vfmadd231ps	%ymm13, %ymm15, %ymm7   # ymm7 = (ymm15 * ymm13) + ymm7
	vfmadd231ps	%ymm15, %ymm14, %ymm0   # ymm0 = (ymm14 * ymm15) + ymm0
	vmovups	-96(%r13,%rbp,4), %ymm15
	vfmadd231ps	%ymm12, %ymm15, %ymm9   # ymm9 = (ymm15 * ymm12) + ymm9
	vfmadd231ps	%ymm13, %ymm15, %ymm3   # ymm3 = (ymm15 * ymm13) + ymm3
	vfmadd231ps	%ymm15, %ymm14, %ymm1   # ymm1 = (ymm14 * ymm15) + ymm1
	vmovups	-96(%r10,%rbp,4), %ymm15
	vfmadd231ps	%ymm12, %ymm15, %ymm10  # ymm10 = (ymm15 * ymm12) + ymm10
	vfmadd231ps	%ymm13, %ymm15, %ymm6   # ymm6 = (ymm15 * ymm13) + ymm6
	vfmadd231ps	%ymm15, %ymm14, %ymm5   # ymm5 = (ymm14 * ymm15) + ymm5
	vmovups	-96(%rcx,%rbp,4), %ymm15
	vfmadd231ps	%ymm12, %ymm15, %ymm8   # ymm8 = (ymm15 * ymm12) + ymm8
	vfmadd231ps	%ymm13, %ymm15, %ymm2   # ymm2 = (ymm15 * ymm13) + ymm2
	vfmadd231ps	%ymm14, %ymm15, %ymm4   # ymm4 = (ymm15 * ymm14) + ymm4
	vmovups	-64(%rax,%rbp,4), %ymm12
	vmovups	-64(%r8,%rbp,4), %ymm15
	vmovups	-64(%rdi,%rbp,4), %ymm13
	vmovups	-64(%rdx,%rbp,4), %ymm14
	vfmadd231ps	%ymm12, %ymm15, %ymm11  # ymm11 = (ymm15 * ymm12) + ymm11
	vfmadd231ps	%ymm13, %ymm15, %ymm7   # ymm7 = (ymm15 * ymm13) + ymm7
	vfmadd231ps	%ymm15, %ymm14, %ymm0   # ymm0 = (ymm14 * ymm15) + ymm0
	vmovups	-64(%r13,%rbp,4), %ymm15
	vfmadd231ps	%ymm12, %ymm15, %ymm9   # ymm9 = (ymm15 * ymm12) + ymm9
	vfmadd231ps	%ymm13, %ymm15, %ymm3   # ymm3 = (ymm15 * ymm13) + ymm3
	vfmadd231ps	%ymm15, %ymm14, %ymm1   # ymm1 = (ymm14 * ymm15) + ymm1
	vmovups	-64(%r10,%rbp,4), %ymm15
	vfmadd231ps	%ymm12, %ymm15, %ymm10  # ymm10 = (ymm15 * ymm12) + ymm10
	vfmadd231ps	%ymm13, %ymm15, %ymm6   # ymm6 = (ymm15 * ymm13) + ymm6
	vfmadd231ps	%ymm15, %ymm14, %ymm5   # ymm5 = (ymm14 * ymm15) + ymm5
	vmovups	-64(%rcx,%rbp,4), %ymm15
	vfmadd231ps	%ymm12, %ymm15, %ymm8   # ymm8 = (ymm15 * ymm12) + ymm8
	vfmadd231ps	%ymm13, %ymm15, %ymm2   # ymm2 = (ymm15 * ymm13) + ymm2
	vfmadd231ps	%ymm14, %ymm15, %ymm4   # ymm4 = (ymm15 * ymm14) + ymm4
	vmovups	-32(%rax,%rbp,4), %ymm12
	vmovups	-32(%r8,%rbp,4), %ymm15
	vmovups	-32(%rdi,%rbp,4), %ymm13
	vmovups	-32(%rdx,%rbp,4), %ymm14
	vfmadd231ps	%ymm12, %ymm15, %ymm11  # ymm11 = (ymm15 * ymm12) + ymm11
	vfmadd231ps	%ymm13, %ymm15, %ymm7   # ymm7 = (ymm15 * ymm13) + ymm7
	vfmadd231ps	%ymm15, %ymm14, %ymm0   # ymm0 = (ymm14 * ymm15) + ymm0
	vmovups	-32(%r13,%rbp,4), %ymm15
	vfmadd231ps	%ymm12, %ymm15, %ymm9   # ymm9 = (ymm15 * ymm12) + ymm9
	vfmadd231ps	%ymm13, %ymm15, %ymm3   # ymm3 = (ymm15 * ymm13) + ymm3
	vfmadd231ps	%ymm15, %ymm14, %ymm1   # ymm1 = (ymm14 * ymm15) + ymm1
	vmovups	-32(%r10,%rbp,4), %ymm15
	vfmadd231ps	%ymm12, %ymm15, %ymm10  # ymm10 = (ymm15 * ymm12) + ymm10
	vfmadd231ps	%ymm13, %ymm15, %ymm6   # ymm6 = (ymm15 * ymm13) + ymm6
	vfmadd231ps	%ymm15, %ymm14, %ymm5   # ymm5 = (ymm14 * ymm15) + ymm5
	vmovups	-32(%rcx,%rbp,4), %ymm15
	vfmadd231ps	%ymm12, %ymm15, %ymm8   # ymm8 = (ymm15 * ymm12) + ymm8
	vfmadd231ps	%ymm13, %ymm15, %ymm2   # ymm2 = (ymm15 * ymm13) + ymm2
	vfmadd231ps	%ymm14, %ymm15, %ymm4   # ymm4 = (ymm15 * ymm14) + ymm4
	vmovups	(%rax,%rbp,4), %ymm12
	vmovups	(%r8,%rbp,4), %ymm15
	vmovups	(%rdi,%rbp,4), %ymm13
	vmovups	(%rdx,%rbp,4), %ymm14
	vfmadd231ps	%ymm12, %ymm15, %ymm11  # ymm11 = (ymm15 * ymm12) + ymm11
	vfmadd231ps	%ymm13, %ymm15, %ymm7   # ymm7 = (ymm15 * ymm13) + ymm7
	vfmadd231ps	%ymm15, %ymm14, %ymm0   # ymm0 = (ymm14 * ymm15) + ymm0
	vmovups	(%r13,%rbp,4), %ymm15
	vfmadd231ps	%ymm12, %ymm15, %ymm9   # ymm9 = (ymm15 * ymm12) + ymm9
	vfmadd231ps	%ymm13, %ymm15, %ymm3   # ymm3 = (ymm15 * ymm13) + ymm3
	vfmadd231ps	%ymm15, %ymm14, %ymm1   # ymm1 = (ymm14 * ymm15) + ymm1
	vmovups	(%r10,%rbp,4), %ymm15
	vfmadd231ps	%ymm12, %ymm15, %ymm10  # ymm10 = (ymm15 * ymm12) + ymm10
	vfmadd231ps	%ymm13, %ymm15, %ymm6   # ymm6 = (ymm15 * ymm13) + ymm6
	vfmadd231ps	%ymm15, %ymm14, %ymm5   # ymm5 = (ymm14 * ymm15) + ymm5
	vmovups	(%rcx,%rbp,4), %ymm15
	addq	$64, %rbp
	addq	$-8, %rsi
	vfmadd231ps	%ymm12, %ymm15, %ymm8   # ymm8 = (ymm15 * ymm12) + ymm8
	vfmadd231ps	%ymm13, %ymm15, %ymm2   # ymm2 = (ymm15 * ymm13) + ymm2
	vfmadd231ps	%ymm14, %ymm15, %ymm4   # ymm4 = (ymm15 * ymm14) + ymm4
	jne	.LBB1_34
.LBB1_35:                               #   in Loop: Header=BB1_25 Depth=3
	leaq	1(%r12), %rcx
	leaq	2(%r12), %rsi
	movq	%rcx, 56(%rsp)                  # 8-byte Spill
	movq	%rsi, 64(%rsp)                  # 8-byte Spill
	cmpq	$0, 312(%rsp)                   # 8-byte Folded Reload
	je	.LBB1_36
# %bb.37:                               #   in Loop: Header=BB1_25 Depth=3
	movq	%r12, 136(%rsp)                 # 8-byte Spill
	movq	288(%rsp), %r9                  # 8-byte Reload
	movq	304(%rsp), %rcx                 # 8-byte Reload
	movq	296(%rsp), %rsi                 # 8-byte Reload
	movq	(%rsp), %r11                    # 8-byte Reload
	movq	208(%rsp), %r12                 # 8-byte Reload
	leaq	(%r9,%rbp,4), %r15
	movq	32(%rsp), %r9                   # 8-byte Reload
	leaq	(%rcx,%rbp,4), %rbx
	leaq	(%rsi,%rbp,4), %r14
	movq	112(%rsp), %rsi                 # 8-byte Reload
	leaq	(%r9,%rbp,4), %rcx
	movq	24(%rsp), %r9                   # 8-byte Reload
	leaq	(%rsi,%rbp,4), %rsi
	leaq	(%r9,%rbp,4), %r9
	leaq	(%r11,%rbp,4), %rbp
	xorl	%r11d, %r11d
	.p2align	4, 0x90
.LBB1_38:                               #   Parent Loop BB1_9 Depth=1
                                        #     Parent Loop BB1_23 Depth=2
                                        #       Parent Loop BB1_25 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	vmovups	(%rbp,%r11), %ymm12
	vmovups	(%rsi,%r11), %ymm15
	vmovups	(%r9,%r11), %ymm13
	vmovups	(%rcx,%r11), %ymm14
	vfmadd231ps	%ymm12, %ymm15, %ymm11  # ymm11 = (ymm15 * ymm12) + ymm11
	vfmadd231ps	%ymm13, %ymm15, %ymm7   # ymm7 = (ymm15 * ymm13) + ymm7
	vfmadd231ps	%ymm15, %ymm14, %ymm0   # ymm0 = (ymm14 * ymm15) + ymm0
	vmovups	(%r15,%r11), %ymm15
	vfmadd231ps	%ymm12, %ymm15, %ymm9   # ymm9 = (ymm15 * ymm12) + ymm9
	vfmadd231ps	%ymm13, %ymm15, %ymm3   # ymm3 = (ymm15 * ymm13) + ymm3
	vfmadd231ps	%ymm15, %ymm14, %ymm1   # ymm1 = (ymm14 * ymm15) + ymm1
	vmovups	(%r14,%r11), %ymm15
	vfmadd231ps	%ymm12, %ymm15, %ymm10  # ymm10 = (ymm15 * ymm12) + ymm10
	vfmadd231ps	%ymm13, %ymm15, %ymm6   # ymm6 = (ymm15 * ymm13) + ymm6
	vfmadd231ps	%ymm15, %ymm14, %ymm5   # ymm5 = (ymm14 * ymm15) + ymm5
	vmovups	(%rbx,%r11), %ymm15
	addq	$32, %r11
	vfmadd231ps	%ymm12, %ymm15, %ymm8   # ymm8 = (ymm15 * ymm12) + ymm8
	vfmadd231ps	%ymm13, %ymm15, %ymm2   # ymm2 = (ymm15 * ymm13) + ymm2
	vfmadd231ps	%ymm14, %ymm15, %ymm4   # ymm4 = (ymm15 * ymm14) + ymm4
	cmpq	%r11, %r12
	jne	.LBB1_38
# %bb.39:                               #   in Loop: Header=BB1_25 Depth=3
	movq	136(%rsp), %r12                 # 8-byte Reload
	movq	(%rsp), %r11                    # 8-byte Reload
	movq	32(%rsp), %r14                  # 8-byte Reload
	movq	24(%rsp), %rbp                  # 8-byte Reload
	movq	280(%rsp), %r15                 # 8-byte Reload
	movq	64(%rsp), %r9                   # 8-byte Reload
	movq	56(%rsp), %rbx                  # 8-byte Reload
	jmp	.LBB1_40
	.p2align	4, 0x90
.LBB1_36:                               #   in Loop: Header=BB1_25 Depth=3
	movq	%rbx, %r15
	movq	24(%rsp), %rbp                  # 8-byte Reload
	movq	64(%rsp), %r9                   # 8-byte Reload
	movq	56(%rsp), %rbx                  # 8-byte Reload
	movq	32(%rsp), %r14                  # 8-byte Reload
	movq	(%rsp), %r11                    # 8-byte Reload
	jmp	.LBB1_40
	.p2align	4, 0x90
.LBB1_27:                               #   in Loop: Header=BB1_23 Depth=2
	cmpq	16(%rsp), %r12                  # 8-byte Folded Reload
	jge	.LBB1_51
# %bb.28:                               #   in Loop: Header=BB1_23 Depth=2
	movq	8(%rsp), %rdx                   # 8-byte Reload
	movq	%r12, %r8
	movq	40(%rdx), %r14
	movq	48(%rdx), %r13
	movq	8(%rdx), %rax
	movq	16(%rdx), %rcx
	leaq	1(%r12), %rdx
	movq	%r14, %rbp
	movq	%r14, %r10
	movq	%r14, %r11
	imulq	96(%rsp), %rbp                  # 8-byte Folded Reload
	imulq	88(%rsp), %r10                  # 8-byte Folded Reload
	imulq	80(%rsp), %r11                  # 8-byte Folded Reload
	imulq	104(%rsp), %r14                 # 8-byte Folded Reload
	imulq	%r13, %rdx
	imulq	%r13, %r8
	shlq	$3, %r13
	leaq	224(%rcx,%rdx,4), %rsi
	leaq	224(%rcx,%r8,4), %rdi
	leaq	(%rcx,%rdx,4), %rdx
	leaq	(%rcx,%r8,4), %rcx
	leaq	224(%rax,%rbp), %r9
	leaq	224(%rax,%r10), %rbx
	leaq	224(%rax,%r11), %r15
	leaq	224(%rax,%r14), %r8
	addq	%rax, %rbp
	addq	%rax, %r10
	addq	%rax, %r11
	addq	%rax, %r14
	movq	%rbp, 128(%rsp)                 # 8-byte Spill
	movq	%r10, 120(%rsp)                 # 8-byte Spill
	movq	%r11, 112(%rsp)                 # 8-byte Spill
	movq	%r14, 56(%rsp)                  # 8-byte Spill
	movq	%rcx, %rbp
	movq	48(%rsp), %rax                  # 8-byte Reload
	movq	184(%rsp), %rcx                 # 8-byte Reload
	orq	192(%rsp), %rax                 # 8-byte Folded Reload
	leaq	(%rcx,%rax,4), %rax
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movq	%r13, 64(%rsp)                  # 8-byte Spill
	jmp	.LBB1_29
	.p2align	4, 0x90
.LBB1_30:                               #   in Loop: Header=BB1_29 Depth=3
	leaq	1(%r12), %r14
	vxorps	%xmm0, %xmm0, %xmm0
	vxorps	%xmm1, %xmm1, %xmm1
	vxorps	%xmm4, %xmm4, %xmm4
	vxorps	%xmm6, %xmm6, %xmm6
	vxorps	%xmm5, %xmm5, %xmm5
	vxorps	%xmm7, %xmm7, %xmm7
	vxorps	%xmm3, %xmm3, %xmm3
	vxorps	%xmm2, %xmm2, %xmm2
.LBB1_50:                               #   in Loop: Header=BB1_29 Depth=3
	vextractf128	$1, %ymm7, %xmm8
	vextractf128	$1, %ymm5, %xmm9
	movq	224(%rsp), %rcx                 # 8-byte Reload
	movq	32(%rsp), %r10                  # 8-byte Reload
	addq	%r13, %rsi
	addq	%r13, %rdi
	addq	%r13, %rdx
	addq	%r13, %rbp
	vaddps	%xmm7, %xmm8, %xmm7
	vaddps	%xmm5, %xmm9, %xmm5
	movq	%rcx, %rax
	imulq	%r12, %rax
	imulq	%rcx, %r14
	addq	$2, %r12
	vshufpd	$1, %xmm7, %xmm7, %xmm8         # xmm8 = xmm7[1,0]
	vshufpd	$1, %xmm5, %xmm5, %xmm9         # xmm9 = xmm5[1,0]
	vaddps	%xmm7, %xmm8, %xmm7
	vextractf128	$1, %ymm6, %xmm8
	vaddps	%xmm5, %xmm9, %xmm5
	vinsertps	$76, %xmm7, %xmm5, %xmm9 # xmm9 = xmm7[1],xmm5[1],zero,zero
	vunpcklps	%xmm5, %xmm7, %xmm5     # xmm5 = xmm7[0],xmm5[0],xmm7[1],xmm5[1]
	vaddps	%xmm6, %xmm8, %xmm6
	vshufpd	$1, %xmm6, %xmm6, %xmm8         # xmm8 = xmm6[1,0]
	vaddps	%xmm6, %xmm8, %xmm6
	vextractf128	$1, %ymm4, %xmm8
	vaddps	%xmm4, %xmm8, %xmm4
	vshufpd	$1, %xmm4, %xmm4, %xmm8         # xmm8 = xmm4[1,0]
	vaddps	%xmm4, %xmm8, %xmm4
	vshufps	$85, %xmm6, %xmm4, %xmm8        # xmm8 = xmm4[1,1],xmm6[1,1]
	vmovlhps	%xmm6, %xmm4, %xmm4             # xmm4 = xmm4[0],xmm6[0]
	vshufps	$36, %xmm4, %xmm5, %xmm4        # xmm4 = xmm5[0,1],xmm4[2,0]
	vshufps	$36, %xmm8, %xmm9, %xmm8        # xmm8 = xmm9[0,1],xmm8[2,0]
	vextractf128	$1, %ymm0, %xmm5
	vaddps	%xmm4, %xmm8, %xmm4
	vmovups	%xmm4, (%r10,%rax,4)
	vextractf128	$1, %ymm1, %xmm4
	vaddps	%xmm0, %xmm5, %xmm0
	vshufpd	$1, %xmm0, %xmm0, %xmm5         # xmm5 = xmm0[1,0]
	vaddps	%xmm1, %xmm4, %xmm1
	vaddps	%xmm5, %xmm0, %xmm0
	vshufpd	$1, %xmm1, %xmm1, %xmm4         # xmm4 = xmm1[1,0]
	vaddps	%xmm4, %xmm1, %xmm1
	vextractf128	$1, %ymm3, %xmm4
	vinsertps	$76, %xmm1, %xmm0, %xmm5 # xmm5 = xmm1[1],xmm0[1],zero,zero
	vunpcklps	%xmm0, %xmm1, %xmm0     # xmm0 = xmm1[0],xmm0[0],xmm1[1],xmm0[1]
	vaddps	%xmm3, %xmm4, %xmm3
	vshufpd	$1, %xmm3, %xmm3, %xmm4         # xmm4 = xmm3[1,0]
	vaddps	%xmm4, %xmm3, %xmm3
	vextractf128	$1, %ymm2, %xmm4
	vaddps	%xmm2, %xmm4, %xmm2
	vshufpd	$1, %xmm2, %xmm2, %xmm4         # xmm4 = xmm2[1,0]
	vaddps	%xmm4, %xmm2, %xmm2
	vshufps	$85, %xmm3, %xmm2, %xmm4        # xmm4 = xmm2[1,1],xmm3[1,1]
	vmovlhps	%xmm3, %xmm2, %xmm2             # xmm2 = xmm2[0],xmm3[0]
	vshufps	$36, %xmm4, %xmm5, %xmm4        # xmm4 = xmm5[0,1],xmm4[2,0]
	vshufps	$36, %xmm2, %xmm0, %xmm0        # xmm0 = xmm0[0,1],xmm2[2,0]
	vaddps	%xmm4, %xmm0, %xmm0
	vmovups	%xmm0, (%r10,%r14,4)
	cmpq	16(%rsp), %r12                  # 8-byte Folded Reload
	jge	.LBB1_51
.LBB1_29:                               #   Parent Loop BB1_9 Depth=1
                                        #     Parent Loop BB1_23 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB1_44 Depth 4
                                        #         Child Loop BB1_47 Depth 4
	cmpq	$0, 72(%rsp)                    # 8-byte Folded Reload
	jle	.LBB1_30
# %bb.41:                               #   in Loop: Header=BB1_29 Depth=3
	movq	%rbp, (%rsp)                    # 8-byte Spill
	cmpq	$57, 72(%rsp)                   # 8-byte Folded Reload
	jae	.LBB1_43
# %bb.42:                               #   in Loop: Header=BB1_29 Depth=3
	vxorps	%xmm0, %xmm0, %xmm0
	vxorps	%xmm1, %xmm1, %xmm1
	vxorps	%xmm4, %xmm4, %xmm4
	vxorps	%xmm6, %xmm6, %xmm6
	vxorps	%xmm5, %xmm5, %xmm5
	vxorps	%xmm7, %xmm7, %xmm7
	vxorps	%xmm3, %xmm3, %xmm3
	vxorps	%xmm2, %xmm2, %xmm2
	xorl	%ebp, %ebp
	jmp	.LBB1_45
	.p2align	4, 0x90
.LBB1_43:                               #   in Loop: Header=BB1_29 Depth=3
	movq	200(%rsp), %r14                 # 8-byte Reload
	vxorps	%xmm0, %xmm0, %xmm0
	vxorps	%xmm1, %xmm1, %xmm1
	vxorps	%xmm4, %xmm4, %xmm4
	vxorps	%xmm6, %xmm6, %xmm6
	vxorps	%xmm5, %xmm5, %xmm5
	vxorps	%xmm7, %xmm7, %xmm7
	vxorps	%xmm3, %xmm3, %xmm3
	vxorps	%xmm2, %xmm2, %xmm2
	xorl	%ebp, %ebp
	.p2align	4, 0x90
.LBB1_44:                               #   Parent Loop BB1_9 Depth=1
                                        #     Parent Loop BB1_23 Depth=2
                                        #       Parent Loop BB1_29 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	vmovups	-224(%rdi,%rbp,4), %ymm8
	vmovups	-224(%r8,%rbp,4), %ymm9
	vmovups	-224(%rsi,%rbp,4), %ymm10
	vmovups	-96(%r8,%rbp,4), %ymm12
	vmovups	-96(%r15,%rbp,4), %ymm11
	vfmadd231ps	%ymm8, %ymm9, %ymm7     # ymm7 = (ymm9 * ymm8) + ymm7
	vfmadd231ps	%ymm9, %ymm10, %ymm1    # ymm1 = (ymm10 * ymm9) + ymm1
	vmovups	-224(%r15,%rbp,4), %ymm9
	vfmadd231ps	%ymm8, %ymm9, %ymm5     # ymm5 = (ymm9 * ymm8) + ymm5
	vfmadd231ps	%ymm9, %ymm10, %ymm0    # ymm0 = (ymm10 * ymm9) + ymm0
	vmovups	-224(%rbx,%rbp,4), %ymm9
	vfmadd231ps	%ymm8, %ymm9, %ymm6     # ymm6 = (ymm9 * ymm8) + ymm6
	vfmadd231ps	%ymm9, %ymm10, %ymm3    # ymm3 = (ymm10 * ymm9) + ymm3
	vmovups	-224(%r9,%rbp,4), %ymm9
	vfmadd231ps	%ymm8, %ymm9, %ymm4     # ymm4 = (ymm9 * ymm8) + ymm4
	vfmadd231ps	%ymm10, %ymm9, %ymm2    # ymm2 = (ymm9 * ymm10) + ymm2
	vmovups	-192(%rdi,%rbp,4), %ymm8
	vmovups	-192(%r8,%rbp,4), %ymm9
	vmovups	-192(%rsi,%rbp,4), %ymm10
	vfmadd231ps	%ymm8, %ymm9, %ymm7     # ymm7 = (ymm9 * ymm8) + ymm7
	vfmadd231ps	%ymm9, %ymm10, %ymm1    # ymm1 = (ymm10 * ymm9) + ymm1
	vmovups	-192(%r15,%rbp,4), %ymm9
	vfmadd231ps	%ymm8, %ymm9, %ymm5     # ymm5 = (ymm9 * ymm8) + ymm5
	vfmadd231ps	%ymm9, %ymm10, %ymm0    # ymm0 = (ymm10 * ymm9) + ymm0
	vmovups	-192(%rbx,%rbp,4), %ymm9
	vfmadd231ps	%ymm8, %ymm9, %ymm6     # ymm6 = (ymm9 * ymm8) + ymm6
	vfmadd231ps	%ymm9, %ymm10, %ymm3    # ymm3 = (ymm10 * ymm9) + ymm3
	vmovups	-192(%r9,%rbp,4), %ymm9
	vfmadd231ps	%ymm8, %ymm9, %ymm4     # ymm4 = (ymm9 * ymm8) + ymm4
	vfmadd231ps	%ymm10, %ymm9, %ymm2    # ymm2 = (ymm9 * ymm10) + ymm2
	vmovups	-160(%rdi,%rbp,4), %ymm8
	vmovups	-160(%r8,%rbp,4), %ymm9
	vmovups	-160(%rsi,%rbp,4), %ymm10
	vfmadd231ps	%ymm8, %ymm9, %ymm7     # ymm7 = (ymm9 * ymm8) + ymm7
	vfmadd231ps	%ymm9, %ymm10, %ymm1    # ymm1 = (ymm10 * ymm9) + ymm1
	vmovups	-160(%r15,%rbp,4), %ymm9
	vfmadd231ps	%ymm8, %ymm9, %ymm5     # ymm5 = (ymm9 * ymm8) + ymm5
	vfmadd231ps	%ymm9, %ymm10, %ymm0    # ymm0 = (ymm10 * ymm9) + ymm0
	vmovups	-160(%rbx,%rbp,4), %ymm9
	vfmadd231ps	%ymm8, %ymm9, %ymm6     # ymm6 = (ymm9 * ymm8) + ymm6
	vfmadd231ps	%ymm9, %ymm10, %ymm3    # ymm3 = (ymm10 * ymm9) + ymm3
	vmovups	-160(%r9,%rbp,4), %ymm9
	vfmadd231ps	%ymm8, %ymm9, %ymm4     # ymm4 = (ymm9 * ymm8) + ymm4
	vfmadd231ps	%ymm10, %ymm9, %ymm2    # ymm2 = (ymm9 * ymm10) + ymm2
	vmovups	-128(%rdi,%rbp,4), %ymm8
	vmovups	-128(%r8,%rbp,4), %ymm9
	vmovups	-128(%rsi,%rbp,4), %ymm10
	vfmadd231ps	%ymm8, %ymm9, %ymm7     # ymm7 = (ymm9 * ymm8) + ymm7
	vfmadd231ps	%ymm9, %ymm10, %ymm1    # ymm1 = (ymm10 * ymm9) + ymm1
	vmovups	-128(%r15,%rbp,4), %ymm9
	vfmadd231ps	%ymm8, %ymm9, %ymm5     # ymm5 = (ymm9 * ymm8) + ymm5
	vfmadd231ps	%ymm9, %ymm10, %ymm0    # ymm0 = (ymm10 * ymm9) + ymm0
	vmovups	-128(%rbx,%rbp,4), %ymm9
	vfmadd231ps	%ymm8, %ymm9, %ymm6     # ymm6 = (ymm9 * ymm8) + ymm6
	vfmadd231ps	%ymm9, %ymm10, %ymm3    # ymm3 = (ymm10 * ymm9) + ymm3
	vmovups	-128(%r9,%rbp,4), %ymm9
	vfmadd231ps	%ymm8, %ymm9, %ymm4     # ymm4 = (ymm9 * ymm8) + ymm4
	vmovups	-96(%rdi,%rbp,4), %ymm8
	vfmadd231ps	%ymm10, %ymm9, %ymm2    # ymm2 = (ymm9 * ymm10) + ymm2
	vmovups	-96(%rsi,%rbp,4), %ymm9
	vfmadd231ps	%ymm8, %ymm12, %ymm7    # ymm7 = (ymm12 * ymm8) + ymm7
	vfmadd231ps	%ymm12, %ymm9, %ymm1    # ymm1 = (ymm9 * ymm12) + ymm1
	vmovups	-96(%rbx,%rbp,4), %ymm12
	vfmadd231ps	%ymm8, %ymm11, %ymm5    # ymm5 = (ymm11 * ymm8) + ymm5
	vfmadd231ps	%ymm11, %ymm9, %ymm0    # ymm0 = (ymm9 * ymm11) + ymm0
	vmovups	-96(%r9,%rbp,4), %ymm11
	vfmadd231ps	%ymm8, %ymm12, %ymm6    # ymm6 = (ymm12 * ymm8) + ymm6
	vfmadd231ps	%ymm12, %ymm9, %ymm3    # ymm3 = (ymm9 * ymm12) + ymm3
	vfmadd231ps	%ymm8, %ymm11, %ymm4    # ymm4 = (ymm11 * ymm8) + ymm4
	vfmadd231ps	%ymm9, %ymm11, %ymm2    # ymm2 = (ymm11 * ymm9) + ymm2
	vmovups	-64(%rdi,%rbp,4), %ymm8
	vmovups	-64(%r8,%rbp,4), %ymm12
	vmovups	-64(%rsi,%rbp,4), %ymm9
	vmovups	-64(%r15,%rbp,4), %ymm11
	vfmadd231ps	%ymm8, %ymm12, %ymm7    # ymm7 = (ymm12 * ymm8) + ymm7
	vfmadd231ps	%ymm12, %ymm9, %ymm1    # ymm1 = (ymm9 * ymm12) + ymm1
	vmovups	-64(%rbx,%rbp,4), %ymm12
	vfmadd231ps	%ymm8, %ymm11, %ymm5    # ymm5 = (ymm11 * ymm8) + ymm5
	vfmadd231ps	%ymm11, %ymm9, %ymm0    # ymm0 = (ymm9 * ymm11) + ymm0
	vmovups	-64(%r9,%rbp,4), %ymm11
	vfmadd231ps	%ymm8, %ymm12, %ymm6    # ymm6 = (ymm12 * ymm8) + ymm6
	vfmadd231ps	%ymm12, %ymm9, %ymm3    # ymm3 = (ymm9 * ymm12) + ymm3
	vfmadd231ps	%ymm8, %ymm11, %ymm4    # ymm4 = (ymm11 * ymm8) + ymm4
	vfmadd231ps	%ymm9, %ymm11, %ymm2    # ymm2 = (ymm11 * ymm9) + ymm2
	vmovups	-32(%rdi,%rbp,4), %ymm8
	vmovups	-32(%r8,%rbp,4), %ymm12
	vmovups	-32(%rsi,%rbp,4), %ymm9
	vmovups	-32(%r15,%rbp,4), %ymm11
	vfmadd231ps	%ymm8, %ymm12, %ymm7    # ymm7 = (ymm12 * ymm8) + ymm7
	vfmadd231ps	%ymm12, %ymm9, %ymm1    # ymm1 = (ymm9 * ymm12) + ymm1
	vmovups	-32(%rbx,%rbp,4), %ymm12
	vfmadd231ps	%ymm8, %ymm11, %ymm5    # ymm5 = (ymm11 * ymm8) + ymm5
	vfmadd231ps	%ymm11, %ymm9, %ymm0    # ymm0 = (ymm9 * ymm11) + ymm0
	vmovups	-32(%r9,%rbp,4), %ymm11
	vfmadd231ps	%ymm8, %ymm12, %ymm6    # ymm6 = (ymm12 * ymm8) + ymm6
	vfmadd231ps	%ymm12, %ymm9, %ymm3    # ymm3 = (ymm9 * ymm12) + ymm3
	vfmadd231ps	%ymm8, %ymm11, %ymm4    # ymm4 = (ymm11 * ymm8) + ymm4
	vfmadd231ps	%ymm9, %ymm11, %ymm2    # ymm2 = (ymm11 * ymm9) + ymm2
	vmovups	(%rdi,%rbp,4), %ymm8
	vmovups	(%r8,%rbp,4), %ymm12
	vmovups	(%rsi,%rbp,4), %ymm9
	vmovups	(%r15,%rbp,4), %ymm11
	vfmadd231ps	%ymm8, %ymm12, %ymm7    # ymm7 = (ymm12 * ymm8) + ymm7
	vfmadd231ps	%ymm12, %ymm9, %ymm1    # ymm1 = (ymm9 * ymm12) + ymm1
	vmovups	(%rbx,%rbp,4), %ymm12
	vfmadd231ps	%ymm8, %ymm11, %ymm5    # ymm5 = (ymm11 * ymm8) + ymm5
	vfmadd231ps	%ymm11, %ymm9, %ymm0    # ymm0 = (ymm9 * ymm11) + ymm0
	vmovups	(%r9,%rbp,4), %ymm11
	addq	$64, %rbp
	addq	$-8, %r14
	vfmadd231ps	%ymm8, %ymm12, %ymm6    # ymm6 = (ymm12 * ymm8) + ymm6
	vfmadd231ps	%ymm12, %ymm9, %ymm3    # ymm3 = (ymm9 * ymm12) + ymm3
	vfmadd231ps	%ymm8, %ymm11, %ymm4    # ymm4 = (ymm11 * ymm8) + ymm4
	vfmadd231ps	%ymm9, %ymm11, %ymm2    # ymm2 = (ymm11 * ymm9) + ymm2
	jne	.LBB1_44
.LBB1_45:                               #   in Loop: Header=BB1_29 Depth=3
	testb	$7, 320(%rsp)                   # 1-byte Folded Reload
	leaq	1(%r12), %r14
	je	.LBB1_49
# %bb.46:                               #   in Loop: Header=BB1_29 Depth=3
	movq	%r12, 136(%rsp)                 # 8-byte Spill
	movq	%rdx, %r12
	movq	120(%rsp), %r10                 # 8-byte Reload
	movq	128(%rsp), %rax                 # 8-byte Reload
	movq	112(%rsp), %r13                 # 8-byte Reload
	leaq	(%r10,%rbp,4), %r11
	movq	56(%rsp), %r10                  # 8-byte Reload
	movq	%r12, 24(%rsp)                  # 8-byte Spill
	leaq	(%rax,%rbp,4), %rcx
	leaq	(%r12,%rbp,4), %rax
	leaq	(%r13,%rbp,4), %rdx
	movq	(%rsp), %r12                    # 8-byte Reload
	movq	208(%rsp), %r13                 # 8-byte Reload
	leaq	(%r10,%rbp,4), %r10
	leaq	(%r12,%rbp,4), %rbp
	xorl	%r12d, %r12d
	.p2align	4, 0x90
.LBB1_47:                               #   Parent Loop BB1_9 Depth=1
                                        #     Parent Loop BB1_23 Depth=2
                                        #       Parent Loop BB1_29 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	vmovups	(%rbp,%r12), %ymm8
	vmovups	(%rdx,%r12), %ymm11
	vmovups	(%rax,%r12), %ymm9
	vmovups	(%r10,%r12), %ymm10
	vmovups	(%r11,%r12), %ymm12
	vfmadd231ps	%ymm8, %ymm11, %ymm5    # ymm5 = (ymm11 * ymm8) + ymm5
	vfmadd231ps	%ymm11, %ymm9, %ymm0    # ymm0 = (ymm9 * ymm11) + ymm0
	vmovups	(%rcx,%r12), %ymm11
	vfmadd231ps	%ymm8, %ymm10, %ymm7    # ymm7 = (ymm10 * ymm8) + ymm7
	vfmadd231ps	%ymm10, %ymm9, %ymm1    # ymm1 = (ymm9 * ymm10) + ymm1
	vfmadd231ps	%ymm8, %ymm12, %ymm6    # ymm6 = (ymm12 * ymm8) + ymm6
	vfmadd231ps	%ymm12, %ymm9, %ymm3    # ymm3 = (ymm9 * ymm12) + ymm3
	addq	$32, %r12
	vfmadd231ps	%ymm8, %ymm11, %ymm4    # ymm4 = (ymm11 * ymm8) + ymm4
	vfmadd231ps	%ymm9, %ymm11, %ymm2    # ymm2 = (ymm11 * ymm9) + ymm2
	cmpq	%r12, %r13
	jne	.LBB1_47
# %bb.48:                               #   in Loop: Header=BB1_29 Depth=3
	movq	136(%rsp), %r12                 # 8-byte Reload
	movq	64(%rsp), %r13                  # 8-byte Reload
	movq	24(%rsp), %rdx                  # 8-byte Reload
.LBB1_49:                               #   in Loop: Header=BB1_29 Depth=3
	movq	(%rsp), %rbp                    # 8-byte Reload
	jmp	.LBB1_50
	.p2align	4, 0x90
.LBB1_51:                               #   in Loop: Header=BB1_23 Depth=2
	movq	216(%rsp), %r9                  # 8-byte Reload
	cmpq	16(%rsp), %r12                  # 8-byte Folded Reload
	jne	.LBB1_52
# %bb.22:                               #   in Loop: Header=BB1_23 Depth=2
	movq	48(%rsp), %rcx                  # 8-byte Reload
	addq	$16, 96(%rsp)                   # 8-byte Folded Spill
	addq	$16, 88(%rsp)                   # 8-byte Folded Spill
	addq	$16, 80(%rsp)                   # 8-byte Folded Spill
	addq	$16, 104(%rsp)                  # 8-byte Folded Spill
	leaq	4(%rcx), %rax
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	cmpq	$11, %rcx
	jbe	.LBB1_23
# %bb.53:                               #   in Loop: Header=BB1_9 Depth=1
	movq	8(%rsp), %rbx                   # 8-byte Reload
	movl	$1, %esi
	movq	(%rbx), %rax
	movq	24(%rax), %rdi
	vzeroupper
	callq	ggml_threadpool_chunk_add@PLT
	movq	240(%rsp), %rsi                 # 8-byte Reload
	movq	168(%rsp), %rdi                 # 8-byte Reload
	movq	160(%rsp), %r8                  # 8-byte Reload
	movq	232(%rsp), %r9                  # 8-byte Reload
	cltq
	cmpq	%rax, 176(%rsp)                 # 8-byte Folded Reload
	jg	.LBB1_9
# %bb.54:
	movq	(%rbx), %rcx
.LBB1_55:
	movq	24(%rcx), %rdi
	addq	$328, %rsp                      # imm = 0x148
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	jmp	ggml_barrier@PLT                # TAILCALL
.LBB1_52:
	.cfi_def_cfa_offset 384
	leaq	.L.str(%rip), %rdi
	leaq	.L.str.1(%rip), %rdx
	leaq	.L.str.4(%rip), %rcx
	movl	$624, %esi                      # imm = 0x270
	xorl	%eax, %eax
	vzeroupper
	callq	ggml_abort@PLT
.LBB1_56:
	leaq	.L.str(%rip), %rdi
	leaq	.L.str.1(%rip), %rdx
	leaq	.L.str.2(%rip), %rcx
	movl	$584, %esi                      # imm = 0x248
	xorl	%eax, %eax
	callq	ggml_abort@PLT
.LBB1_57:
	leaq	.L.str(%rip), %rdi
	leaq	.L.str.1(%rip), %rdx
	leaq	.L.str.3(%rip), %rcx
	movl	$596, %esi                      # imm = 0x254
	xorl	%eax, %eax
	callq	ggml_abort@PLT
.Lfunc_end1:
	.size	_ZN12_GLOBAL__N_18tinyBLASILi8EDv8_fS1_fffE4gemmILi4ELi3ELi4EEEvlll, .Lfunc_end1-_ZN12_GLOBAL__N_18tinyBLASILi8EDv8_fS1_fffE4gemmILi4ELi3ELi4EEEvlll
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _ZN12_GLOBAL__N_18tinyBLASILi8EDv8_fS1_fffE4gemmILi4ELi2ELi4EEEvlll
	.type	_ZN12_GLOBAL__N_18tinyBLASILi8EDv8_fS1_fffE4gemmILi4ELi2ELi4EEEvlll,@function
