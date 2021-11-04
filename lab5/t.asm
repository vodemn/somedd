    .mmregs	
	.def _c_int00
	.text
		
_c_int00:
	xor A, A
	xor B, B
	rsbx OVM
	ssbx frct
	stm #Mem, AR5
     
main:
    andm #0xFFE0, *(0x001D)
    xor A, A
	xor B, B
	stm sin, AR2
	stm cos, AR3
	stm delta, AR1
	st #N, AR6		
	st #0, *AR5+
	rpt #k-1		
	add *AR1, A;a=a+ar1
	nop
	stl A, AR4 
	stl A, *AR1

main_loop:
	ld *AR1, 16, A
	exp A
	st T, *AR2
	ld *(buf1), B
	sub *AR2, B 
	stl B, AR7
	add #1, B
	neg B
	stl B, T
	nop
	norm A
	nop
	ld *(buf2), B
	stl B, T
	nop
	norm A
	nop 
	stl A, *AR2 ; 
	mpy *AR2, *AR2, A
	ld #32767, B
	sub A, -16, B ;cos
	stl B, *AR3

get_angle:
	mpy *AR3, *AR2, A ;cos(a(n))*sin(a(n))
	mpy *AR2, *AR2, B ;sin^2(a(n))
	stl A, -15, *AR2 ;(2cos*sin)
	ld #32767, A
	sub B, -15, A ;1-2*sin(a)^2
	stl A, *AR3
	nop
	banz get_angle, *AR7-
	nop
	nop
	mvdd *AR2, *AR5+
	ld *AR1, A
	add AR4, A
	stl A, *AR1
	banz main_loop, *AR6-
	nop
	nop

cascade1:
	nop
	orm #1, *(0x001D)
	stm N, AR2
	ldm AR5, A
	add #-225, A
	stlm A, AR5
	stm #dnm2, AR3
	rsbx OVM
	nop
	rptz A, 2
	stl A, *AR3-
	nop

	st #1482, *(a0)
	st #1489, *(a1)
	st #1482, *(a2)
	st #21540, *(b1)
	st #-25519, *(b2)
	nop
	call filter
	nop

cascade2:
	nop
	stm N, AR2
	ldm AR5, A
	add #-225, A
	stlm A, AR5
	stm #dnm2, AR3
	nop
	rptz A, 1
	stl A, *AR3-
	nop

	st #27423, *(a0)
	st #-27558, *(a1)
	st #27423, *(a2)
	st #-25519, *(b1)
	st #-21282, *(b2)
	nop
	call filter
	nop

cascade3:
	stm N, AR2
	ldm AR5, A
	add #-225, A
	stlm A, AR5
	stm #dnm2, AR3
	nop
	rptz A, 1
	stl A, *AR3-
	nop

	st #5719, *(a0)
	st #0, *(a1)
	st #-5719, *(a2)
	st #-20976, *(b1)
	st #-16411, *(b2)
	nop
	call filter
	nop
	nop		;return
	
	ldm AR5, A
	add #-1, A
	stlm A, AR5
	addm #-1, *(Gar)
	mvdm Gar, AR2	
	
	addm #1, *(buf1)
	addm #1, *(buf2)
	st #14, *(delta)
	nop
	banz main, *AR2
	nop		;return
	nop
	nop

filter:
	ld *AR5, 16, A ; dn
	stm #dnm2, AR3
	rpt #1
	macp *AR3- , b2, A
	sth A, *AR3
	nop

	Xor A, A ; yn
	nop
	stm #dnm2, AR3
	macd *AR3-, a2, A
	mpy a2, B
	add B, A
	macd *AR3-, a1, A
	mpy a1, B
	add B, A
	mpy a1, B
	add B, A
	macd *AR3, a0, A
	mpy a0, B
	add B, A
	sth A, *AR5+
	banz filter, *AR2-
	ret


N 		.set 224
k      	.set 8
Mem     .set 500h

		.data
sin 	.word 1
cos 	.word 1
delta   .word 19	
Gar		.word 3
buf1    .word 11
buf2    .word -9
a2		.word 1482 ;/2	
a1		.word 1489 ;/3	
a0		.word 1482 ;/2	
p1		.word 0	
b2		.word -25519		
b1		.word 21540	
p2		.word 0
dn      .word 0
dnm1    .word 0	
dnm2    .word 0
p3		.word 0
