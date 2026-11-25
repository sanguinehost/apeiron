
target/release/examples/prefix_solve:     file format elf64-x86-64


Disassembly of section .text:

000000000010ab30 <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2>:
  10ab30:	push   %rbp
  10ab31:	mov    %rsp,%rbp
  10ab34:	push   %r15
  10ab36:	push   %r14
  10ab38:	push   %r13
  10ab3a:	push   %r12
  10ab3c:	push   %rbx
  10ab3d:	and    $0xffffffffffffffe0,%rsp
  10ab41:	sub    $0x180,%rsp
  10ab48:	mov    %r8,0x8(%rsp)
  10ab4d:	mov    %rdx,0x10(%rsp)
  10ab52:	mov    %rsi,0x30(%rsp)
  10ab57:	mov    %rdi,%r10
  10ab5a:	vpxor  %xmm6,%xmm6,%xmm6
  10ab5e:	vmovdqa %ymm6,0x140(%rsp)
  10ab67:	vmovdqa %ymm6,0x120(%rsp)
  10ab70:	vmovdqa %ymm6,0x100(%rsp)
  10ab79:	vmovdqa %ymm6,0xe0(%rsp)
  10ab82:	vmovdqa %ymm6,0xc0(%rsp)
  10ab8b:	vmovdqa %ymm6,0xa0(%rsp)
  10ab94:	vmovdqa %ymm6,0x80(%rsp)
  10ab9d:	vmovdqa %ymm6,0x60(%rsp)
  10aba3:	mov    %rcx,%rsi
  10aba6:	shr    $0x5,%rsi
  10abaa:	vpxor  %xmm7,%xmm7,%xmm7
  10abae:	vpxor  %xmm0,%xmm0,%xmm0
  10abb2:	vpxor  %xmm1,%xmm1,%xmm1
  10abb6:	vpxor  %xmm2,%xmm2,%xmm2
  10abba:	vpxor  %xmm3,%xmm3,%xmm3
  10abbe:	vpxor  %xmm4,%xmm4,%xmm4
  10abc2:	vpxor  %xmm5,%xmm5,%xmm5
  10abc6:	mov    %rdi,0x18(%rsp)
  10abcb:	mov    %rcx,0x38(%rsp)
  10abd0:	je     10ae04 <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0x2d4>
  10abd6:	cmpq   $0x0,0x10(%rsp)
  10abdc:	je     10ad9d <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0x26d>
  10abe2:	mov    0x18(%rsp),%r8
  10abe7:	mov    (%r8),%rax
  10abea:	mov    %rax,0x28(%rsp)
  10abef:	mov    0x20(%r8),%rax
  10abf3:	mov    %rax,0x20(%rsp)
  10abf8:	mov    0x38(%r8),%rax
  10abfc:	mov    %rax,0x58(%rsp)
  10ac01:	mov    0x58(%r8),%rax
  10ac05:	mov    %rax,0x50(%rsp)
  10ac0a:	mov    0x70(%r8),%rax
  10ac0e:	mov    %rax,0x48(%rsp)
  10ac13:	mov    0x90(%r8),%r11
  10ac1a:	mov    0xa8(%r8),%r14
  10ac21:	mov    0xc8(%r8),%r15
  10ac28:	mov    0xe0(%r8),%r12
  10ac2f:	mov    0x100(%r8),%r13
  10ac36:	mov    0x118(%r8),%r10
  10ac3d:	mov    0x138(%r8),%rax
  10ac44:	mov    0x150(%r8),%rdi
  10ac4b:	vmovdqa -0xe54f3(%rip),%ymm8        # 25760 <anon.02cae0c252322f20e1441761f4796d6b.8.llvm.4092852103517430976+0x1c0>
  10ac53:	vmovdqa -0xe573b(%rip),%ymm9        # 25520 <anon.f7cf31c838997622f5af1efb3220822e.3.llvm.11822430256438140027+0x29c>
  10ac5b:	vpbroadcastd -0xe5edc(%rip),%ymm10        # 24d88 <anon.f7cf31c838997622f5af1efb3220822e.0.llvm.11822430256438140027+0x331>
  10ac64:	vmovdqa -0xe560c(%rip),%ymm11        # 25660 <anon.02cae0c252322f20e1441761f4796d6b.8.llvm.4092852103517430976+0xc0>
  10ac6c:	vpbroadcastw -0xe43f9(%rip),%ymm12        # 2687c <anon.db0080edd14254defd0a65f22bd63a29.2.llvm.15563377919032525938+0x98>
  10ac75:	mov    0x170(%r8),%rcx
  10ac7c:	mov    0x188(%r8),%rdx
  10ac83:	mov    0x1a8(%r8),%r9
  10ac8a:	mov    0x10(%rsp),%r8
  10ac8f:	lea    0x0(,%r8,4),%r8
  10ac97:	mov    %r8,0x40(%rsp)
  10ac9c:	xor    %r8d,%r8d
  10ac9f:	nop
  10aca0:	mov    0x30(%rsp),%rbx
  10aca5:	vpbroadcastd (%rbx,%r8,1),%ymm13
  10acab:	vpsrlvd %ymm8,%ymm13,%ymm13
  10acb0:	vpshufb %ymm9,%ymm13,%ymm13
  10acb5:	vpand  %ymm10,%ymm13,%ymm13
  10acba:	vpshufb %ymm13,%ymm11,%ymm13
  10acbf:	mov    0x28(%rsp),%rbx
  10acc4:	vpmaddubsw (%rbx,%r8,8),%ymm13,%ymm14
  10acca:	vpmaddwd %ymm12,%ymm14,%ymm14
  10accf:	mov    0x20(%rsp),%rbx
  10acd4:	vpsllvd (%rbx,%r8,8),%ymm14,%ymm14
  10acda:	vpaddd %ymm7,%ymm14,%ymm7
  10acde:	mov    0x58(%rsp),%rbx
  10ace3:	vpmaddubsw (%rbx,%r8,8),%ymm13,%ymm14
  10ace9:	vpmaddwd %ymm12,%ymm14,%ymm14
  10acee:	mov    0x50(%rsp),%rbx
  10acf3:	vpsllvd (%rbx,%r8,8),%ymm14,%ymm14
  10acf9:	mov    0x48(%rsp),%rbx
  10acfe:	vpmaddubsw (%rbx,%r8,8),%ymm13,%ymm15
  10ad04:	vpaddd %ymm6,%ymm14,%ymm6
  10ad08:	vpmaddwd %ymm12,%ymm15,%ymm14
  10ad0d:	vpsllvd (%r11,%r8,8),%ymm14,%ymm14
  10ad13:	vpaddd %ymm5,%ymm14,%ymm5
  10ad17:	vpmaddubsw (%r14,%r8,8),%ymm13,%ymm14
  10ad1d:	vpmaddwd %ymm12,%ymm14,%ymm14
  10ad22:	vpsllvd (%r15,%r8,8),%ymm14,%ymm14
  10ad28:	vpmaddubsw (%r12,%r8,8),%ymm13,%ymm15
  10ad2e:	vpaddd %ymm4,%ymm14,%ymm4
  10ad32:	vpmaddwd %ymm12,%ymm15,%ymm14
  10ad37:	vpsllvd 0x0(%r13,%r8,8),%ymm14,%ymm14
  10ad3e:	vpaddd %ymm3,%ymm14,%ymm3
  10ad42:	vpmaddubsw (%r10,%r8,8),%ymm13,%ymm14
  10ad48:	vpmaddwd %ymm12,%ymm14,%ymm14
  10ad4d:	vpsllvd (%rax,%r8,8),%ymm14,%ymm14
  10ad53:	vpmaddubsw (%rdi,%r8,8),%ymm13,%ymm15
  10ad59:	vpaddd %ymm2,%ymm14,%ymm2
  10ad5d:	vpmaddwd %ymm12,%ymm15,%ymm14
  10ad62:	vpsllvd (%rcx,%r8,8),%ymm14,%ymm14
  10ad68:	vpaddd %ymm1,%ymm14,%ymm1
  10ad6c:	vpmaddubsw (%rdx,%r8,8),%ymm13,%ymm13
  10ad72:	vpmaddwd %ymm12,%ymm13,%ymm13
  10ad77:	vpsllvd (%r9,%r8,8),%ymm13,%ymm13
  10ad7d:	vpaddd %ymm0,%ymm13,%ymm0
  10ad81:	lea    -0x4(,%rsi,4),%rbx
  10ad89:	cmp    %r8,%rbx
  10ad8c:	je     10adb5 <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0x285>
  10ad8e:	add    $0x4,%r8
  10ad92:	cmp    %r8,0x40(%rsp)
  10ad97:	jne    10aca0 <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0x170>
  10ad9d:	lea    0xf6e44(%rip),%rdx        # 201be8 <anon.7cb005ddfd1a844be01ecb1c973d7dd1.73.llvm.14048548577563402891+0x338>
  10ada4:	mov    0x10(%rsp),%rdi
  10ada9:	mov    %rdi,%rsi
  10adac:	vzeroupper
  10adaf:	call   *0xff293(%rip)        # 20a048 <_DYNAMIC+0x3a0>
  10adb5:	vmovdqa %ymm7,0x60(%rsp)
  10adbb:	vmovdqa %ymm6,0x80(%rsp)
  10adc4:	vmovdqa %ymm5,0xa0(%rsp)
  10adcd:	vmovdqa %ymm4,0xc0(%rsp)
  10add6:	vmovdqa %ymm3,0xe0(%rsp)
  10addf:	vmovdqa %ymm2,0x100(%rsp)
  10ade8:	vmovdqa %ymm1,0x120(%rsp)
  10adf1:	vmovdqa %ymm0,0x140(%rsp)
  10adfa:	mov    0x18(%rsp),%r10
  10adff:	mov    0x38(%rsp),%rcx
  10ae04:	mov    %rcx,%rdi
  10ae07:	and    $0xffffffffffffffe0,%rdi
  10ae0b:	cmp    %rcx,%rdi
  10ae0e:	jne    10b199 <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0x669>
  10ae14:	movabs $0x8000000000000000,%rdx
  10ae1e:	vpmovsxdq %xmm7,%ymm8
  10ae23:	vextracti128 $0x1,%ymm7,%xmm7
  10ae29:	vpmovsxdq %xmm7,%ymm7
  10ae2e:	vpaddq %ymm7,%ymm8,%ymm7
  10ae32:	vextracti128 $0x1,%ymm7,%xmm8
  10ae38:	vpaddq %xmm7,%xmm8,%xmm7
  10ae3c:	vpshufd $0xee,%xmm7,%xmm8
  10ae41:	vpaddq %xmm7,%xmm8,%xmm7
  10ae45:	vmovq  %xmm7,%rsi
  10ae4a:	mov    %rsi,%rdi
  10ae4d:	sar    $0x3f,%rdi
  10ae51:	movzbl 0x30(%r10),%ecx
  10ae56:	shld   %cl,%rsi,%rdi
  10ae5a:	shl    %cl,%rsi
  10ae5d:	xor    %eax,%eax
  10ae5f:	test   $0x40,%cl
  10ae62:	cmovne %rsi,%rdi
  10ae66:	mov    $0x0,%ecx
  10ae6b:	cmove  %rsi,%rcx
  10ae6f:	mov    %rcx,%rsi
  10ae72:	add    %rdx,%rsi
  10ae75:	adc    $0xffffffffffffffff,%rdi
  10ae79:	cmp    $0xffffffffffffffff,%rdi
  10ae7d:	jne    10b76f <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc3f>
  10ae83:	mov    $0xffffffffffffffff,%rsi
  10ae8a:	mov    0x8(%rsp),%rax
  10ae8f:	mov    %rcx,(%rax)
  10ae92:	vpmovsxdq %xmm6,%ymm7
  10ae97:	vextracti128 $0x1,%ymm6,%xmm6
  10ae9d:	vpmovsxdq %xmm6,%ymm6
  10aea2:	vpaddq %ymm6,%ymm7,%ymm6
  10aea6:	vextracti128 $0x1,%ymm6,%xmm7
  10aeac:	vpaddq %xmm6,%xmm7,%xmm6
  10aeb0:	vpshufd $0xee,%xmm6,%xmm7
  10aeb5:	vpaddq %xmm7,%xmm6,%xmm6
  10aeb9:	vmovq  %xmm6,%rdi
  10aebe:	mov    %rdi,%r8
  10aec1:	sar    $0x3f,%r8
  10aec5:	movzbl 0x68(%r10),%ecx
  10aeca:	shld   %cl,%rdi,%r8
  10aece:	shl    %cl,%rdi
  10aed1:	xor    %eax,%eax
  10aed3:	test   $0x40,%cl
  10aed6:	cmovne %rdi,%r8
  10aeda:	mov    $0x0,%ecx
  10aedf:	cmove  %rdi,%rcx
  10aee3:	mov    %rcx,%rdi
  10aee6:	add    %rdx,%rdi
  10aee9:	adc    %rsi,%r8
  10aeec:	cmp    $0xffffffffffffffff,%r8
  10aef0:	jne    10b76f <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc3f>
  10aef6:	mov    0x8(%rsp),%rax
  10aefb:	mov    %rcx,0x8(%rax)
  10aeff:	vpmovsxdq %xmm5,%ymm6
  10af04:	vextracti128 $0x1,%ymm5,%xmm5
  10af0a:	vpmovsxdq %xmm5,%ymm5
  10af0f:	vpaddq %ymm5,%ymm6,%ymm5
  10af13:	vextracti128 $0x1,%ymm5,%xmm6
  10af19:	vpaddq %xmm5,%xmm6,%xmm5
  10af1d:	vpshufd $0xee,%xmm5,%xmm6
  10af22:	vpaddq %xmm6,%xmm5,%xmm5
  10af26:	vmovq  %xmm5,%rdi
  10af2b:	mov    %rdi,%r8
  10af2e:	sar    $0x3f,%r8
  10af32:	movzbl 0xa0(%r10),%ecx
  10af3a:	shld   %cl,%rdi,%r8
  10af3e:	shl    %cl,%rdi
  10af41:	xor    %eax,%eax
  10af43:	test   $0x40,%cl
  10af46:	cmovne %rdi,%r8
  10af4a:	mov    $0x0,%ecx
  10af4f:	cmove  %rdi,%rcx
  10af53:	mov    %rcx,%rdi
  10af56:	add    %rdx,%rdi
  10af59:	adc    %rsi,%r8
  10af5c:	cmp    $0xffffffffffffffff,%r8
  10af60:	jne    10b76f <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc3f>
  10af66:	mov    0x8(%rsp),%rax
  10af6b:	mov    %rcx,0x10(%rax)
  10af6f:	vpmovsxdq %xmm4,%ymm5
  10af74:	vextracti128 $0x1,%ymm4,%xmm4
  10af7a:	vpmovsxdq %xmm4,%ymm4
  10af7f:	vpaddq %ymm4,%ymm5,%ymm4
  10af83:	vextracti128 $0x1,%ymm4,%xmm5
  10af89:	vpaddq %xmm4,%xmm5,%xmm4
  10af8d:	vpshufd $0xee,%xmm4,%xmm5
  10af92:	vpaddq %xmm5,%xmm4,%xmm4
  10af96:	vmovq  %xmm4,%rdi
  10af9b:	mov    %rdi,%r8
  10af9e:	sar    $0x3f,%r8
  10afa2:	movzbl 0xd8(%r10),%ecx
  10afaa:	shld   %cl,%rdi,%r8
  10afae:	shl    %cl,%rdi
  10afb1:	xor    %eax,%eax
  10afb3:	test   $0x40,%cl
  10afb6:	cmovne %rdi,%r8
  10afba:	mov    $0x0,%ecx
  10afbf:	cmove  %rdi,%rcx
  10afc3:	mov    %rcx,%rdi
  10afc6:	add    %rdx,%rdi
  10afc9:	adc    %rsi,%r8
  10afcc:	cmp    $0xffffffffffffffff,%r8
  10afd0:	jne    10b76f <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc3f>
  10afd6:	mov    0x8(%rsp),%rax
  10afdb:	mov    %rcx,0x18(%rax)
  10afdf:	vpmovsxdq %xmm3,%ymm4
  10afe4:	vextracti128 $0x1,%ymm3,%xmm3
  10afea:	vpmovsxdq %xmm3,%ymm3
  10afef:	vpaddq %ymm3,%ymm4,%ymm3
  10aff3:	vextracti128 $0x1,%ymm3,%xmm4
  10aff9:	vpaddq %xmm3,%xmm4,%xmm3
  10affd:	vpshufd $0xee,%xmm3,%xmm4
  10b002:	vpaddq %xmm4,%xmm3,%xmm3
  10b006:	vmovq  %xmm3,%rdi
  10b00b:	mov    %rdi,%r8
  10b00e:	sar    $0x3f,%r8
  10b012:	movzbl 0x110(%r10),%ecx
  10b01a:	shld   %cl,%rdi,%r8
  10b01e:	shl    %cl,%rdi
  10b021:	xor    %eax,%eax
  10b023:	test   $0x40,%cl
  10b026:	cmovne %rdi,%r8
  10b02a:	mov    $0x0,%ecx
  10b02f:	cmove  %rdi,%rcx
  10b033:	mov    %rcx,%rdi
  10b036:	add    %rdx,%rdi
  10b039:	adc    %rsi,%r8
  10b03c:	cmp    $0xffffffffffffffff,%r8
  10b040:	jne    10b76f <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc3f>
  10b046:	mov    0x8(%rsp),%rax
  10b04b:	mov    %rcx,0x20(%rax)
  10b04f:	vpmovsxdq %xmm2,%ymm3
  10b054:	vextracti128 $0x1,%ymm2,%xmm2
  10b05a:	vpmovsxdq %xmm2,%ymm2
  10b05f:	vpaddq %ymm2,%ymm3,%ymm2
  10b063:	vextracti128 $0x1,%ymm2,%xmm3
  10b069:	vpaddq %xmm2,%xmm3,%xmm2
  10b06d:	vpshufd $0xee,%xmm2,%xmm3
  10b072:	vpaddq %xmm3,%xmm2,%xmm2
  10b076:	vmovq  %xmm2,%rdi
  10b07b:	mov    %rdi,%r8
  10b07e:	sar    $0x3f,%r8
  10b082:	movzbl 0x148(%r10),%ecx
  10b08a:	shld   %cl,%rdi,%r8
  10b08e:	shl    %cl,%rdi
  10b091:	xor    %eax,%eax
  10b093:	test   $0x40,%cl
  10b096:	cmovne %rdi,%r8
  10b09a:	mov    $0x0,%ecx
  10b09f:	cmove  %rdi,%rcx
  10b0a3:	mov    %rcx,%rdi
  10b0a6:	add    %rdx,%rdi
  10b0a9:	adc    %rsi,%r8
  10b0ac:	cmp    $0xffffffffffffffff,%r8
  10b0b0:	jne    10b76f <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc3f>
  10b0b6:	mov    0x8(%rsp),%rax
  10b0bb:	mov    %rcx,0x28(%rax)
  10b0bf:	vpmovsxdq %xmm1,%ymm2
  10b0c4:	vextracti128 $0x1,%ymm1,%xmm1
  10b0ca:	vpmovsxdq %xmm1,%ymm1
  10b0cf:	vpaddq %ymm1,%ymm2,%ymm1
  10b0d3:	vextracti128 $0x1,%ymm1,%xmm2
  10b0d9:	vpaddq %xmm1,%xmm2,%xmm1
  10b0dd:	vpshufd $0xee,%xmm1,%xmm2
  10b0e2:	vpaddq %xmm2,%xmm1,%xmm1
  10b0e6:	vmovq  %xmm1,%r8
  10b0eb:	mov    %r8,%r9
  10b0ee:	sar    $0x3f,%r9
  10b0f2:	movzbl 0x180(%r10),%ecx
  10b0fa:	shld   %cl,%r8,%r9
  10b0fe:	shl    %cl,%r8
  10b101:	xor    %eax,%eax
  10b103:	test   $0x40,%cl
  10b106:	cmovne %r8,%r9
  10b10a:	mov    $0x0,%edi
  10b10f:	cmove  %r8,%rdi
  10b113:	mov    %rdi,%rcx
  10b116:	add    %rdx,%rcx
  10b119:	adc    %rsi,%r9
  10b11c:	cmp    $0xffffffffffffffff,%r9
  10b120:	jne    10b76f <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc3f>
  10b126:	vpmovsxdq %xmm0,%ymm1
  10b12b:	vextracti128 $0x1,%ymm0,%xmm0
  10b131:	vpmovsxdq %xmm0,%ymm0
  10b136:	vpaddq %ymm0,%ymm1,%ymm0
  10b13a:	vextracti128 $0x1,%ymm0,%xmm1
  10b140:	vpaddq %xmm0,%xmm1,%xmm0
  10b144:	vpshufd $0xee,%xmm0,%xmm1
  10b149:	vpaddq %xmm1,%xmm0,%xmm0
  10b14d:	vmovq  %xmm0,%r8
  10b152:	mov    %r8,%r9
  10b155:	sar    $0x3f,%r9
  10b159:	movzbl 0x1b8(%r10),%ecx
  10b161:	shld   %cl,%r8,%r9
  10b165:	shl    %cl,%r8
  10b168:	xor    %eax,%eax
  10b16a:	test   $0x40,%cl
  10b16d:	cmovne %r8,%r9
  10b171:	mov    0x8(%rsp),%rcx
  10b176:	mov    %rdi,0x30(%rcx)
  10b17a:	mov    $0x0,%r11d
  10b180:	cmove  %r8,%r11
  10b184:	add    %r11,%rdx
  10b187:	adc    %rsi,%r9
  10b18a:	cmp    $0xffffffffffffffff,%r9
  10b18e:	je     10b781 <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc51>
  10b194:	jmp    10b76f <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc3f>
  10b199:	cmp    0x10(%rsp),%rsi
  10b19e:	ja     10b78e <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc5e>
  10b1a4:	mov    %ecx,%r15d
  10b1a7:	and    $0x1f,%r15d
  10b1ab:	je     10b32d <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0x7fd>
  10b1b1:	xor    %r9d,%r9d
  10b1b4:	mov    %rdi,0x20(%rsp)
  10b1b9:	nopl   0x0(%rax)
  10b1c0:	mov    %r9,%rax
  10b1c3:	shl    $0x5,%rax
  10b1c7:	vpmovsxdq 0x60(%rsp,%rax,1),%ymm0
  10b1ce:	vpmovsxdq 0x70(%rsp,%rax,1),%ymm1
  10b1d5:	vpaddq %ymm1,%ymm0,%ymm0
  10b1d9:	vextracti128 $0x1,%ymm0,%xmm1
  10b1df:	vpaddq %xmm0,%xmm1,%xmm0
  10b1e3:	vpshufd $0xee,%xmm0,%xmm1
  10b1e8:	vpaddq %xmm1,%xmm0,%xmm0
  10b1ec:	vmovq  %xmm0,%rdx
  10b1f1:	mov    %rdx,%r11
  10b1f4:	sar    $0x3f,%r11
  10b1f8:	imul   $0x38,%r9,%rax
  10b1fc:	movzbl 0x30(%r10,%rax,1),%ecx
  10b202:	shld   %cl,%rdx,%r11
  10b206:	shl    %cl,%rdx
  10b209:	test   $0x40,%cl
  10b20c:	cmovne %rdx,%r11
  10b210:	mov    $0x0,%r8d
  10b216:	cmove  %rdx,%r8
  10b21a:	mov    0x8(%r10,%rax,1),%rdx
  10b21f:	mov    %rdx,%rcx
  10b222:	sub    %rdi,%rcx
  10b225:	jb     10b7bb <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc8b>
  10b22b:	add    %r10,%rax
  10b22e:	mov    0x18(%rax),%rdx
  10b232:	cmp    %rdx,%rsi
  10b235:	ja     10b7ce <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc9e>
  10b23b:	cmp    %rsi,0x10(%rsp)
  10b240:	je     10b7e4 <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xcb4>
  10b246:	mov    %r9,0x28(%rsp)
  10b24b:	cmp    %rsi,%rdx
  10b24e:	je     10b7f8 <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xcc8>
  10b254:	mov    (%rax),%r13
  10b257:	add    %rdi,%r13
  10b25a:	mov    0x10(%rax),%r9
  10b25e:	xor    %edi,%edi
  10b260:	xor    %r10d,%r10d
  10b263:	xor    %r14d,%r14d
  10b266:	jmp    10b281 <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0x751>
  10b268:	nopl   0x0(%rax,%rax,1)
  10b270:	xor    %eax,%eax
  10b272:	imulq  (%r9,%rsi,8)
  10b276:	add    %rax,%rdi
  10b279:	adc    %rdx,%r10
  10b27c:	cmp    %r15,%r14
  10b27f:	jae    10b2e0 <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0x7b0>
  10b281:	lea    0x20(%r14),%rdx
  10b285:	cmp    %rdx,%r15
  10b288:	cmovb  %r15,%rdx
  10b28c:	cmp    %r14,%r15
  10b28f:	jbe    10b270 <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0x740>
  10b291:	mov    0x30(%rsp),%rax
  10b296:	mov    (%rax,%rsi,4),%r12d
  10b29a:	xor    %eax,%eax
  10b29c:	jmp    10b2a8 <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0x778>
  10b29e:	xchg   %ax,%ax
  10b2a0:	inc    %r14
  10b2a3:	cmp    %r14,%rdx
  10b2a6:	je     10b2d0 <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0x7a0>
  10b2a8:	bt     %r14d,%r12d
  10b2ac:	jae    10b2a0 <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0x770>
  10b2ae:	cmp    %rcx,%r14
  10b2b1:	jae    10b9d3 <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xea3>
  10b2b7:	movsbq 0x0(%r13,%r14,1),%rbx
  10b2bd:	add    %rbx,%rax
  10b2c0:	jmp    10b2a0 <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0x770>
  10b2c2:	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  10b2d0:	mov    %rdx,%r14
  10b2d3:	jmp    10b272 <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0x742>
  10b2d5:	data16 cs nopw 0x0(%rax,%rax,1)
  10b2e0:	add    %r8,%rdi
  10b2e3:	adc    %r11,%r10
  10b2e6:	mov    %rdi,%rax
  10b2e9:	movabs $0x8000000000000000,%rcx
  10b2f3:	add    %rcx,%rax
  10b2f6:	adc    $0x0,%r10
  10b2fa:	sete   %al
  10b2fd:	jne    10b76f <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc3f>
  10b303:	mov    0x8(%rsp),%rcx
  10b308:	mov    0x28(%rsp),%r9
  10b30d:	mov    %rdi,(%rcx,%r9,8)
  10b311:	inc    %r9
  10b314:	cmp    $0x8,%r9
  10b318:	mov    0x18(%rsp),%r10
  10b31d:	mov    0x20(%rsp),%rdi
  10b322:	jne    10b1c0 <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0x690>
  10b328:	jmp    10b76f <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc3f>
  10b32d:	vpmovsxdq %xmm7,%ymm8
  10b332:	vextracti128 $0x1,%ymm7,%xmm7
  10b338:	vpmovsxdq %xmm7,%ymm7
  10b33d:	vpaddq %ymm7,%ymm8,%ymm7
  10b341:	vextracti128 $0x1,%ymm7,%xmm8
  10b347:	vpaddq %xmm7,%xmm8,%xmm7
  10b34b:	vpshufd $0xee,%xmm7,%xmm8
  10b350:	vpaddq %xmm7,%xmm8,%xmm7
  10b354:	vmovq  %xmm7,%rax
  10b359:	mov    %rax,%r8
  10b35c:	sar    $0x3f,%r8
  10b360:	movzbl 0x30(%r10),%ecx
  10b365:	shld   %cl,%rax,%r8
  10b369:	shl    %cl,%rax
  10b36c:	xor    %r9d,%r9d
  10b36f:	test   $0x40,%cl
  10b372:	cmovne %rax,%r8
  10b376:	cmove  %rax,%r9
  10b37a:	mov    0x8(%r10),%rdx
  10b37e:	cmp    %rdx,%rdi
  10b381:	ja     10b7bb <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc8b>
  10b387:	mov    0x18(%r10),%rdx
  10b38b:	cmp    %rdx,%rsi
  10b38e:	ja     10b7ce <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc9e>
  10b394:	movabs $0x8000000000000000,%rax
  10b39e:	mov    %r9,%rcx
  10b3a1:	add    %rax,%rcx
  10b3a4:	adc    $0xffffffffffffffff,%r8
  10b3a8:	cmp    $0xffffffffffffffff,%r8
  10b3ac:	jne    10b76d <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc3d>
  10b3b2:	mov    0x8(%rsp),%rcx
  10b3b7:	mov    %r9,(%rcx)
  10b3ba:	vpmovsxdq %xmm6,%ymm7
  10b3bf:	vextracti128 $0x1,%ymm6,%xmm6
  10b3c5:	vpmovsxdq %xmm6,%ymm6
  10b3ca:	vpaddq %ymm6,%ymm7,%ymm6
  10b3ce:	vextracti128 $0x1,%ymm6,%xmm7
  10b3d4:	vpaddq %xmm6,%xmm7,%xmm6
  10b3d8:	vpshufd $0xee,%xmm6,%xmm7
  10b3dd:	vpaddq %xmm7,%xmm6,%xmm6
  10b3e1:	vmovq  %xmm6,%rdx
  10b3e6:	mov    %rdx,%r11
  10b3e9:	sar    $0x3f,%r11
  10b3ed:	movzbl 0x68(%r10),%ecx
  10b3f2:	shld   %cl,%rdx,%r11
  10b3f6:	shl    %cl,%rdx
  10b3f9:	xor    %r8d,%r8d
  10b3fc:	test   $0x40,%cl
  10b3ff:	cmovne %rdx,%r11
  10b403:	cmove  %rdx,%r8
  10b407:	mov    0x40(%r10),%rdx
  10b40b:	cmp    %rdx,%rdi
  10b40e:	ja     10b7bb <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc8b>
  10b414:	mov    0x50(%r10),%rdx
  10b418:	cmp    %rdx,%rsi
  10b41b:	ja     10b7ce <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc9e>
  10b421:	mov    $0xffffffffffffffff,%r9
  10b428:	mov    %r8,%rcx
  10b42b:	add    %rax,%rcx
  10b42e:	adc    %r9,%r11
  10b431:	cmp    $0xffffffffffffffff,%r11
  10b435:	jne    10b76d <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc3d>
  10b43b:	mov    0x8(%rsp),%rcx
  10b440:	mov    %r8,0x8(%rcx)
  10b444:	vpmovsxdq %xmm5,%ymm6
  10b449:	vextracti128 $0x1,%ymm5,%xmm5
  10b44f:	vpmovsxdq %xmm5,%ymm5
  10b454:	vpaddq %ymm5,%ymm6,%ymm5
  10b458:	vextracti128 $0x1,%ymm5,%xmm6
  10b45e:	vpaddq %xmm5,%xmm6,%xmm5
  10b462:	vpshufd $0xee,%xmm5,%xmm6
  10b467:	vpaddq %xmm6,%xmm5,%xmm5
  10b46b:	vmovq  %xmm5,%rdx
  10b470:	mov    %rdx,%r11
  10b473:	sar    $0x3f,%r11
  10b477:	movzbl 0xa0(%r10),%ecx
  10b47f:	shld   %cl,%rdx,%r11
  10b483:	shl    %cl,%rdx
  10b486:	xor    %r8d,%r8d
  10b489:	test   $0x40,%cl
  10b48c:	cmovne %rdx,%r11
  10b490:	cmove  %rdx,%r8
  10b494:	mov    0x78(%r10),%rdx
  10b498:	cmp    %rdx,%rdi
  10b49b:	ja     10b7bb <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc8b>
  10b4a1:	mov    0x88(%r10),%rdx
  10b4a8:	cmp    %rdx,%rsi
  10b4ab:	ja     10b7ce <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc9e>
  10b4b1:	mov    %r8,%rcx
  10b4b4:	add    %rax,%rcx
  10b4b7:	adc    %r9,%r11
  10b4ba:	cmp    $0xffffffffffffffff,%r11
  10b4be:	jne    10b76d <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc3d>
  10b4c4:	mov    0x8(%rsp),%rcx
  10b4c9:	mov    %r8,0x10(%rcx)
  10b4cd:	vpmovsxdq %xmm4,%ymm5
  10b4d2:	vextracti128 $0x1,%ymm4,%xmm4
  10b4d8:	vpmovsxdq %xmm4,%ymm4
  10b4dd:	vpaddq %ymm4,%ymm5,%ymm4
  10b4e1:	vextracti128 $0x1,%ymm4,%xmm5
  10b4e7:	vpaddq %xmm4,%xmm5,%xmm4
  10b4eb:	vpshufd $0xee,%xmm4,%xmm5
  10b4f0:	vpaddq %xmm5,%xmm4,%xmm4
  10b4f4:	vmovq  %xmm4,%rdx
  10b4f9:	mov    %rdx,%r11
  10b4fc:	sar    $0x3f,%r11
  10b500:	movzbl 0xd8(%r10),%ecx
  10b508:	shld   %cl,%rdx,%r11
  10b50c:	shl    %cl,%rdx
  10b50f:	xor    %r8d,%r8d
  10b512:	test   $0x40,%cl
  10b515:	cmovne %rdx,%r11
  10b519:	cmove  %rdx,%r8
  10b51d:	mov    0xb0(%r10),%rdx
  10b524:	cmp    %rdx,%rdi
  10b527:	ja     10b7bb <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc8b>
  10b52d:	mov    0xc0(%r10),%rdx
  10b534:	cmp    %rdx,%rsi
  10b537:	ja     10b7ce <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc9e>
  10b53d:	mov    %r8,%rcx
  10b540:	add    %rax,%rcx
  10b543:	adc    %r9,%r11
  10b546:	cmp    $0xffffffffffffffff,%r11
  10b54a:	jne    10b76d <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc3d>
  10b550:	mov    0x8(%rsp),%rcx
  10b555:	mov    %r8,0x18(%rcx)
  10b559:	vpmovsxdq %xmm3,%ymm4
  10b55e:	vextracti128 $0x1,%ymm3,%xmm3
  10b564:	vpmovsxdq %xmm3,%ymm3
  10b569:	vpaddq %ymm3,%ymm4,%ymm3
  10b56d:	vextracti128 $0x1,%ymm3,%xmm4
  10b573:	vpaddq %xmm3,%xmm4,%xmm3
  10b577:	vpshufd $0xee,%xmm3,%xmm4
  10b57c:	vpaddq %xmm4,%xmm3,%xmm3
  10b580:	vmovq  %xmm3,%rdx
  10b585:	mov    %rdx,%r11
  10b588:	sar    $0x3f,%r11
  10b58c:	movzbl 0x110(%r10),%ecx
  10b594:	shld   %cl,%rdx,%r11
  10b598:	shl    %cl,%rdx
  10b59b:	xor    %r8d,%r8d
  10b59e:	test   $0x40,%cl
  10b5a1:	cmovne %rdx,%r11
  10b5a5:	cmove  %rdx,%r8
  10b5a9:	mov    0xe8(%r10),%rdx
  10b5b0:	cmp    %rdx,%rdi
  10b5b3:	ja     10b7bb <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc8b>
  10b5b9:	mov    0xf8(%r10),%rdx
  10b5c0:	cmp    %rdx,%rsi
  10b5c3:	ja     10b7ce <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc9e>
  10b5c9:	mov    %r8,%rcx
  10b5cc:	add    %rax,%rcx
  10b5cf:	adc    %r9,%r11
  10b5d2:	cmp    $0xffffffffffffffff,%r11
  10b5d6:	jne    10b76d <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc3d>
  10b5dc:	mov    0x8(%rsp),%rcx
  10b5e1:	mov    %r8,0x20(%rcx)
  10b5e5:	vpmovsxdq %xmm2,%ymm3
  10b5ea:	vextracti128 $0x1,%ymm2,%xmm2
  10b5f0:	vpmovsxdq %xmm2,%ymm2
  10b5f5:	vpaddq %ymm2,%ymm3,%ymm2
  10b5f9:	vextracti128 $0x1,%ymm2,%xmm3
  10b5ff:	vpaddq %xmm2,%xmm3,%xmm2
  10b603:	vpshufd $0xee,%xmm2,%xmm3
  10b608:	vpaddq %xmm3,%xmm2,%xmm2
  10b60c:	vmovq  %xmm2,%rdx
  10b611:	mov    %rdx,%r11
  10b614:	sar    $0x3f,%r11
  10b618:	movzbl 0x148(%r10),%ecx
  10b620:	shld   %cl,%rdx,%r11
  10b624:	shl    %cl,%rdx
  10b627:	xor    %r8d,%r8d
  10b62a:	test   $0x40,%cl
  10b62d:	cmovne %rdx,%r11
  10b631:	cmove  %rdx,%r8
  10b635:	mov    0x120(%r10),%rdx
  10b63c:	cmp    %rdx,%rdi
  10b63f:	ja     10b7bb <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc8b>
  10b645:	mov    0x130(%r10),%rdx
  10b64c:	cmp    %rdx,%rsi
  10b64f:	ja     10b7ce <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc9e>
  10b655:	mov    %r8,%rcx
  10b658:	add    %rax,%rcx
  10b65b:	adc    %r9,%r11
  10b65e:	cmp    $0xffffffffffffffff,%r11
  10b662:	jne    10b76d <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc3d>
  10b668:	mov    0x8(%rsp),%rcx
  10b66d:	mov    %r8,0x28(%rcx)
  10b671:	vpmovsxdq %xmm1,%ymm2
  10b676:	vextracti128 $0x1,%ymm1,%xmm1
  10b67c:	vpmovsxdq %xmm1,%ymm1
  10b681:	vpaddq %ymm1,%ymm2,%ymm1
  10b685:	vextracti128 $0x1,%ymm1,%xmm2
  10b68b:	vpaddq %xmm1,%xmm2,%xmm1
  10b68f:	vpshufd $0xee,%xmm1,%xmm2
  10b694:	vpaddq %xmm2,%xmm1,%xmm1
  10b698:	vmovq  %xmm1,%rdx
  10b69d:	mov    %rdx,%r11
  10b6a0:	sar    $0x3f,%r11
  10b6a4:	movzbl 0x180(%r10),%ecx
  10b6ac:	shld   %cl,%rdx,%r11
  10b6b0:	shl    %cl,%rdx
  10b6b3:	xor    %r8d,%r8d
  10b6b6:	test   $0x40,%cl
  10b6b9:	cmovne %rdx,%r11
  10b6bd:	cmove  %rdx,%r8
  10b6c1:	mov    0x158(%r10),%rdx
  10b6c8:	cmp    %rdx,%rdi
  10b6cb:	ja     10b7bb <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc8b>
  10b6d1:	mov    0x168(%r10),%rdx
  10b6d8:	cmp    %rdx,%rsi
  10b6db:	ja     10b7ce <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc9e>
  10b6e1:	mov    %r8,%rcx
  10b6e4:	add    %rax,%rcx
  10b6e7:	adc    %r9,%r11
  10b6ea:	cmp    $0xffffffffffffffff,%r11
  10b6ee:	jne    10b76d <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc3d>
  10b6f0:	mov    0x8(%rsp),%rcx
  10b6f5:	mov    %r8,0x30(%rcx)
  10b6f9:	vpmovsxdq %xmm0,%ymm1
  10b6fe:	vextracti128 $0x1,%ymm0,%xmm0
  10b704:	vpmovsxdq %xmm0,%ymm0
  10b709:	vpaddq %ymm0,%ymm1,%ymm0
  10b70d:	vextracti128 $0x1,%ymm0,%xmm1
  10b713:	vpaddq %xmm0,%xmm1,%xmm0
  10b717:	vpshufd $0xee,%xmm0,%xmm1
  10b71c:	vpaddq %xmm1,%xmm0,%xmm0
  10b720:	vmovq  %xmm0,%rdx
  10b725:	mov    %rdx,%r8
  10b728:	sar    $0x3f,%r8
  10b72c:	movzbl 0x1b8(%r10),%ecx
  10b734:	shld   %cl,%rdx,%r8
  10b738:	shl    %cl,%rdx
  10b73b:	xor    %r11d,%r11d
  10b73e:	test   $0x40,%cl
  10b741:	cmovne %rdx,%r8
  10b745:	cmove  %rdx,%r11
  10b749:	mov    0x190(%r10),%rdx
  10b750:	cmp    %rdx,%rdi
  10b753:	ja     10b7bb <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc8b>
  10b755:	mov    0x1a0(%r10),%rdx
  10b75c:	cmp    %rdx,%rsi
  10b75f:	ja     10b7ce <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc9e>
  10b761:	add    %r11,%rax
  10b764:	adc    %r9,%r8
  10b767:	cmp    $0xffffffffffffffff,%r8
  10b76b:	je     10b781 <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc51>
  10b76d:	xor    %eax,%eax
  10b76f:	lea    -0x28(%rbp),%rsp
  10b773:	pop    %rbx
  10b774:	pop    %r12
  10b776:	pop    %r13
  10b778:	pop    %r14
  10b77a:	pop    %r15
  10b77c:	pop    %rbp
  10b77d:	vzeroupper
  10b780:	ret
  10b781:	mov    0x8(%rsp),%rax
  10b786:	mov    %r11,0x38(%rax)
  10b78a:	mov    $0x1,%al
  10b78c:	jmp    10b76f <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc3f>
  10b78e:	mov    0x8(%r10),%rdx
  10b792:	cmp    %rdx,%rdi
  10b795:	ja     10b7bb <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc8b>
  10b797:	mov    0x18(%r10),%rdx
  10b79b:	cmp    %rdx,%rsi
  10b79e:	ja     10b7ce <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xc9e>
  10b7a0:	lea    0xf6429(%rip),%rcx        # 201bd0 <anon.7cb005ddfd1a844be01ecb1c973d7dd1.73.llvm.14048548577563402891+0x320>
  10b7a7:	mov    %rsi,%rdi
  10b7aa:	mov    0x10(%rsp),%rdx
  10b7af:	mov    %rdx,%rsi
  10b7b2:	vzeroupper
  10b7b5:	call   *0xfe885(%rip)        # 20a040 <_DYNAMIC+0x398>
  10b7bb:	lea    0xf63de(%rip),%rcx        # 201ba0 <anon.7cb005ddfd1a844be01ecb1c973d7dd1.73.llvm.14048548577563402891+0x2f0>
  10b7c2:	mov    %rdx,%rsi
  10b7c5:	vzeroupper
  10b7c8:	call   *0xfe872(%rip)        # 20a040 <_DYNAMIC+0x398>
  10b7ce:	lea    0xf63e3(%rip),%rcx        # 201bb8 <anon.7cb005ddfd1a844be01ecb1c973d7dd1.73.llvm.14048548577563402891+0x308>
  10b7d5:	mov    %rsi,%rdi
  10b7d8:	mov    %rdx,%rsi
  10b7db:	vzeroupper
  10b7de:	call   *0xfe85c(%rip)        # 20a040 <_DYNAMIC+0x398>
  10b7e4:	lea    0xf6325(%rip),%rdx        # 201b10 <anon.7cb005ddfd1a844be01ecb1c973d7dd1.73.llvm.14048548577563402891+0x260>
  10b7eb:	xor    %edi,%edi
  10b7ed:	xor    %esi,%esi
  10b7ef:	vzeroupper
  10b7f2:	call   *0xfe850(%rip)        # 20a048 <_DYNAMIC+0x3a0>
  10b7f8:	mov    0x30(%rsp),%rax
  10b7fd:	mov    (%rax,%rsi,4),%eax
  10b800:	cmp    $0x10,%r15d
  10b804:	jae    10b80e <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xcde>
  10b806:	xor    %r14d,%r14d
  10b809:	jmp    10b9c8 <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xe98>
  10b80e:	mov    0x38(%rsp),%r14
  10b813:	and    $0x18,%r14d
  10b817:	vmovq  %rcx,%xmm0
  10b81c:	vpbroadcastq %xmm0,%ymm6
  10b821:	vmovd  %eax,%xmm0
  10b825:	vpbroadcastd %xmm0,%ymm0
  10b82a:	vmovdqa -0xe6012(%rip),%ymm1        # 25820 <anon.02cae0c252322f20e1441761f4796d6b.8.llvm.4092852103517430976+0x280>
  10b832:	vmovdqa -0xe617a(%rip),%ymm2        # 256c0 <anon.02cae0c252322f20e1441761f4796d6b.8.llvm.4092852103517430976+0x120>
  10b83a:	vmovdqa -0xe62e2(%rip),%ymm3        # 25560 <anon.f7cf31c838997622f5af1efb3220822e.3.llvm.11822430256438140027+0x2dc>
  10b842:	mov    $0x8,%esi
  10b847:	vpbroadcastd -0xe6aa8(%rip),%ymm4        # 24da8 <anon.f7cf31c838997622f5af1efb3220822e.0.llvm.11822430256438140027+0x351>
  10b850:	vpbroadcastq -0xe9fa9(%rip),%ymm5        # 218b0 <GCC_except_table0+0xe48>
  10b859:	vpxor  %ymm5,%ymm6,%ymm6
  10b85d:	vpcmpeqd %ymm7,%ymm7,%ymm7
  10b861:	vpbroadcastq -0xea0d2(%rip),%ymm8        # 21798 <GCC_except_table0+0xd30>
  10b86a:	vpbroadcastd -0xe6a7f(%rip),%ymm9        # 24df4 <anon.f7cf31c838997622f5af1efb3220822e.0.llvm.11822430256438140027+0x39d>
  10b873:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  10b880:	vpand  %ymm4,%ymm3,%ymm10
  10b884:	vpsrlvd %ymm10,%ymm0,%ymm10
  10b889:	vpslld $0x1f,%ymm10,%ymm10
  10b88f:	vpsrad $0x1f,%ymm10,%ymm10
  10b895:	vextracti128 $0x1,%ymm10,%xmm11
  10b89b:	vpackssdw %xmm11,%xmm10,%xmm10
  10b8a0:	vpxor  %ymm5,%ymm2,%ymm11
  10b8a4:	vpcmpgtq %ymm11,%ymm6,%ymm11
  10b8a9:	vpxor  %ymm5,%ymm1,%ymm12
  10b8ad:	vpcmpgtq %ymm12,%ymm6,%ymm12
  10b8b2:	vpackssdw %ymm12,%ymm11,%ymm11
  10b8b7:	vpxor  %ymm7,%ymm11,%ymm11
  10b8bb:	vextracti128 $0x1,%ymm11,%xmm12
  10b8c1:	vpackssdw %xmm12,%xmm11,%xmm11
  10b8c6:	vpshufd $0xd8,%xmm11,%xmm11
  10b8cc:	vpand  %xmm11,%xmm10,%xmm10
  10b8d1:	vpmovzxwd %xmm10,%ymm11
  10b8d6:	vpslld $0x1f,%ymm11,%ymm11
  10b8dc:	vmovmskps %ymm11,%edi
  10b8e1:	lea    -0x8(%rsi),%rdx
  10b8e5:	test   %edi,%edi
  10b8e7:	jne    10b905 <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xdd5>
  10b8e9:	add    %r14,%rsi
  10b8ec:	vpaddq %ymm2,%ymm8,%ymm2
  10b8f0:	vpaddq %ymm1,%ymm8,%ymm1
  10b8f4:	vpaddd %ymm3,%ymm9,%ymm3
  10b8f8:	cmp    $0x10,%rsi
  10b8fc:	mov    %rdx,%rsi
  10b8ff:	jne    10b880 <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xd50>
  10b905:	test   %dil,%dil
  10b908:	je     10b9a7 <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xe77>
  10b90e:	vpshufb -0xea5c7(%rip),%xmm10,%xmm0        # 21350 <GCC_except_table0+0x8e8>
  10b917:	vpsllw $0x7,%xmm0,%xmm0
  10b91c:	vpxor  %xmm1,%xmm1,%xmm1
  10b920:	vpcmpgtb %xmm0,%xmm1,%xmm0
  10b924:	vpand  -0xeaa6c(%rip),%xmm0,%xmm0        # 20ec0 <GCC_except_table0+0x458>
  10b92c:	vpextrb $0x1,%xmm0,%eax
  10b932:	vmovd  %xmm0,%esi
  10b936:	cmp    %al,%sil
  10b939:	cmova  %esi,%eax
  10b93c:	vpextrb $0x2,%xmm0,%esi
  10b942:	cmp    %sil,%al
  10b945:	cmovbe %esi,%eax
  10b948:	vpextrb $0x3,%xmm0,%esi
  10b94e:	cmp    %sil,%al
  10b951:	cmovbe %esi,%eax
  10b954:	vpextrb $0x4,%xmm0,%esi
  10b95a:	cmp    %sil,%al
  10b95d:	cmovbe %esi,%eax
  10b960:	vpextrb $0x5,%xmm0,%esi
  10b966:	cmp    %sil,%al
  10b969:	cmovbe %esi,%eax
  10b96c:	vpextrb $0x6,%xmm0,%esi
  10b972:	cmp    %sil,%al
  10b975:	cmovbe %esi,%eax
  10b978:	vpextrb $0x7,%xmm0,%esi
  10b97e:	cmp    %sil,%al
  10b981:	cmovbe %esi,%eax
  10b984:	mov    $0x8,%sil
  10b987:	sub    %al,%sil
  10b98a:	movzbl %sil,%r14d
  10b98e:	sub    %rdx,%r14
  10b991:	lea    0xf6190(%rip),%rdx        # 201b28 <anon.7cb005ddfd1a844be01ecb1c973d7dd1.73.llvm.14048548577563402891+0x278>
  10b998:	mov    %r14,%rdi
  10b99b:	mov    %rcx,%rsi
  10b99e:	vzeroupper
  10b9a1:	call   *0xfe6a1(%rip)        # 20a048 <_DYNAMIC+0x3a0>
  10b9a7:	cmp    %r14d,%r15d
  10b9aa:	jne    10b9c8 <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xe98>
  10b9ac:	lea    0xf6145(%rip),%rdx        # 201af8 <anon.7cb005ddfd1a844be01ecb1c973d7dd1.73.llvm.14048548577563402891+0x248>
  10b9b3:	xor    %edi,%edi
  10b9b5:	xor    %esi,%esi
  10b9b7:	vzeroupper
  10b9ba:	call   *0xfe688(%rip)        # 20a048 <_DYNAMIC+0x3a0>
  10b9c0:	inc    %r14
  10b9c3:	cmp    %r14,%r15
  10b9c6:	je     10b9ac <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xe7c>
  10b9c8:	bt     %r14d,%eax
  10b9cc:	jae    10b9c0 <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xe90>
  10b9ce:	cmp    %rcx,%r14
  10b9d1:	jb     10b9c0 <_RNvNtNtNtCs7rPN1reWdmT_7apeiron6engine4kern2nq21code_select_row8_avx2+0xe90>
  10b9d3:	lea    0xf614e(%rip),%rdx        # 201b28 <anon.7cb005ddfd1a844be01ecb1c973d7dd1.73.llvm.14048548577563402891+0x278>
  10b9da:	mov    %r14,%rdi
  10b9dd:	mov    %rcx,%rsi
  10b9e0:	vzeroupper
  10b9e3:	call   *0xfe65f(%rip)        # 20a048 <_DYNAMIC+0x3a0>
  10b9e9:	int3
  10b9ea:	int3
  10b9eb:	int3
  10b9ec:	int3
  10b9ed:	int3
  10b9ee:	int3
  10b9ef:	int3
