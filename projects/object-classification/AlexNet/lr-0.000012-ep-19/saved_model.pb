ç
á
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

ú
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%·Ñ8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-0-g3f878cff5b68°

alex_net/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*'
shared_namealex_net/conv2d/kernel

*alex_net/conv2d/kernel/Read/ReadVariableOpReadVariableOpalex_net/conv2d/kernel*&
_output_shapes
:`*
dtype0

alex_net/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*%
shared_namealex_net/conv2d/bias
y
(alex_net/conv2d/bias/Read/ReadVariableOpReadVariableOpalex_net/conv2d/bias*
_output_shapes
:`*
dtype0

"alex_net/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*3
shared_name$"alex_net/batch_normalization/gamma

6alex_net/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp"alex_net/batch_normalization/gamma*
_output_shapes
:`*
dtype0

!alex_net/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*2
shared_name#!alex_net/batch_normalization/beta

5alex_net/batch_normalization/beta/Read/ReadVariableOpReadVariableOp!alex_net/batch_normalization/beta*
_output_shapes
:`*
dtype0
¨
(alex_net/batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*9
shared_name*(alex_net/batch_normalization/moving_mean
¡
<alex_net/batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOp(alex_net/batch_normalization/moving_mean*
_output_shapes
:`*
dtype0
°
,alex_net/batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*=
shared_name.,alex_net/batch_normalization/moving_variance
©
@alex_net/batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp,alex_net/batch_normalization/moving_variance*
_output_shapes
:`*
dtype0

alex_net/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*)
shared_namealex_net/conv2d_1/kernel

,alex_net/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpalex_net/conv2d_1/kernel*'
_output_shapes
:`*
dtype0

alex_net/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namealex_net/conv2d_1/bias
~
*alex_net/conv2d_1/bias/Read/ReadVariableOpReadVariableOpalex_net/conv2d_1/bias*
_output_shapes	
:*
dtype0
¡
$alex_net/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$alex_net/batch_normalization_1/gamma

8alex_net/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp$alex_net/batch_normalization_1/gamma*
_output_shapes	
:*
dtype0

#alex_net/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#alex_net/batch_normalization_1/beta

7alex_net/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp#alex_net/batch_normalization_1/beta*
_output_shapes	
:*
dtype0
­
*alex_net/batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*alex_net/batch_normalization_1/moving_mean
¦
>alex_net/batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp*alex_net/batch_normalization_1/moving_mean*
_output_shapes	
:*
dtype0
µ
.alex_net/batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.alex_net/batch_normalization_1/moving_variance
®
Balex_net/batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp.alex_net/batch_normalization_1/moving_variance*
_output_shapes	
:*
dtype0

alex_net/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namealex_net/conv2d_2/kernel

,alex_net/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpalex_net/conv2d_2/kernel*(
_output_shapes
:*
dtype0

alex_net/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namealex_net/conv2d_2/bias
~
*alex_net/conv2d_2/bias/Read/ReadVariableOpReadVariableOpalex_net/conv2d_2/bias*
_output_shapes	
:*
dtype0
¡
$alex_net/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$alex_net/batch_normalization_2/gamma

8alex_net/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp$alex_net/batch_normalization_2/gamma*
_output_shapes	
:*
dtype0

#alex_net/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#alex_net/batch_normalization_2/beta

7alex_net/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp#alex_net/batch_normalization_2/beta*
_output_shapes	
:*
dtype0
­
*alex_net/batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*alex_net/batch_normalization_2/moving_mean
¦
>alex_net/batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp*alex_net/batch_normalization_2/moving_mean*
_output_shapes	
:*
dtype0
µ
.alex_net/batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.alex_net/batch_normalization_2/moving_variance
®
Balex_net/batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp.alex_net/batch_normalization_2/moving_variance*
_output_shapes	
:*
dtype0

alex_net/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namealex_net/conv2d_3/kernel

,alex_net/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpalex_net/conv2d_3/kernel*(
_output_shapes
:*
dtype0

alex_net/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namealex_net/conv2d_3/bias
~
*alex_net/conv2d_3/bias/Read/ReadVariableOpReadVariableOpalex_net/conv2d_3/bias*
_output_shapes	
:*
dtype0
¡
$alex_net/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$alex_net/batch_normalization_3/gamma

8alex_net/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp$alex_net/batch_normalization_3/gamma*
_output_shapes	
:*
dtype0

#alex_net/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#alex_net/batch_normalization_3/beta

7alex_net/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp#alex_net/batch_normalization_3/beta*
_output_shapes	
:*
dtype0
­
*alex_net/batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*alex_net/batch_normalization_3/moving_mean
¦
>alex_net/batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp*alex_net/batch_normalization_3/moving_mean*
_output_shapes	
:*
dtype0
µ
.alex_net/batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.alex_net/batch_normalization_3/moving_variance
®
Balex_net/batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp.alex_net/batch_normalization_3/moving_variance*
_output_shapes	
:*
dtype0

alex_net/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namealex_net/conv2d_4/kernel

,alex_net/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpalex_net/conv2d_4/kernel*(
_output_shapes
:*
dtype0

alex_net/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namealex_net/conv2d_4/bias
~
*alex_net/conv2d_4/bias/Read/ReadVariableOpReadVariableOpalex_net/conv2d_4/bias*
_output_shapes	
:*
dtype0
¡
$alex_net/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$alex_net/batch_normalization_4/gamma

8alex_net/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp$alex_net/batch_normalization_4/gamma*
_output_shapes	
:*
dtype0

#alex_net/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#alex_net/batch_normalization_4/beta

7alex_net/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp#alex_net/batch_normalization_4/beta*
_output_shapes	
:*
dtype0
­
*alex_net/batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*alex_net/batch_normalization_4/moving_mean
¦
>alex_net/batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp*alex_net/batch_normalization_4/moving_mean*
_output_shapes	
:*
dtype0
µ
.alex_net/batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.alex_net/batch_normalization_4/moving_variance
®
Balex_net/batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp.alex_net/batch_normalization_4/moving_variance*
_output_shapes	
:*
dtype0

alex_net/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	H *&
shared_namealex_net/dense/kernel

)alex_net/dense/kernel/Read/ReadVariableOpReadVariableOpalex_net/dense/kernel*
_output_shapes
:	H *
dtype0
~
alex_net/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_namealex_net/dense/bias
w
'alex_net/dense/bias/Read/ReadVariableOpReadVariableOpalex_net/dense/bias*
_output_shapes
: *
dtype0

alex_net/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *(
shared_namealex_net/dense_1/kernel

+alex_net/dense_1/kernel/Read/ReadVariableOpReadVariableOpalex_net/dense_1/kernel*
_output_shapes

:  *
dtype0

alex_net/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_namealex_net/dense_1/bias
{
)alex_net/dense_1/bias/Read/ReadVariableOpReadVariableOpalex_net/dense_1/bias*
_output_shapes
: *
dtype0

alex_net/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 
*(
shared_namealex_net/dense_2/kernel

+alex_net/dense_2/kernel/Read/ReadVariableOpReadVariableOpalex_net/dense_2/kernel*
_output_shapes

: 
*
dtype0

alex_net/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_namealex_net/dense_2/bias
{
)alex_net/dense_2/bias/Read/ReadVariableOpReadVariableOpalex_net/dense_2/bias*
_output_shapes
:
*
dtype0

NoOpNoOp
¸~
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ó}
valueé}Bæ} Bß}
½
	scale
	conv1
batch_norm1
	pool1
	conv2
batch_norm2
	pool2
	conv3
	batch_norm3
	
conv4
batch_norm4
	conv5
batch_norm5
	pool3
flat

dense1
	drop1

dense2
	drop2

classifier
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*

	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses* 
¦

#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses*
Õ
+axis
	,gamma
-beta
.moving_mean
/moving_variance
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses*

6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses* 
¦

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses*
Õ
Daxis
	Egamma
Fbeta
Gmoving_mean
Hmoving_variance
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses*

O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses* 
¦

Ukernel
Vbias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses*
Õ
]axis
	^gamma
_beta
`moving_mean
amoving_variance
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses*
¦

hkernel
ibias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses*
Õ
paxis
	qgamma
rbeta
smoving_mean
tmoving_variance
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses*
©

{kernel
|bias
}	variables
~trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
à
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
+¡&call_and_return_all_conditional_losses*
¬
¢	variables
£trainable_variables
¤regularization_losses
¥	keras_api
¦_random_generator
§__call__
+¨&call_and_return_all_conditional_losses* 
®
©kernel
	ªbias
«	variables
¬trainable_variables
­regularization_losses
®	keras_api
¯__call__
+°&call_and_return_all_conditional_losses*
¬
±	variables
²trainable_variables
³regularization_losses
´	keras_api
µ_random_generator
¶__call__
+·&call_and_return_all_conditional_losses* 
®
¸kernel
	¹bias
º	variables
»trainable_variables
¼regularization_losses
½	keras_api
¾__call__
+¿&call_and_return_all_conditional_losses*
¤
#0
$1
,2
-3
.4
/5
<6
=7
E8
F9
G10
H11
U12
V13
^14
_15
`16
a17
h18
i19
q20
r21
s22
t23
{24
|25
26
27
28
29
30
31
©32
ª33
¸34
¹35*
Ò
#0
$1
,2
-3
<4
=5
E6
F7
U8
V9
^10
_11
h12
i13
q14
r15
{16
|17
18
19
20
21
©22
ª23
¸24
¹25*
*
À0
Á1
Â2
Ã3
Ä4* 
µ
Ånon_trainable_variables
Ælayers
Çmetrics
 Èlayer_regularization_losses
Élayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Êserving_default* 
* 
* 
* 

Ënon_trainable_variables
Ìlayers
Ímetrics
 Îlayer_regularization_losses
Ïlayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses* 
* 
* 
WQ
VARIABLE_VALUEalex_net/conv2d/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEalex_net/conv2d/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE*

#0
$1*

#0
$1*


À0* 

Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*
* 
* 
* 
hb
VARIABLE_VALUE"alex_net/batch_normalization/gamma,batch_norm1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE!alex_net/batch_normalization/beta+batch_norm1/beta/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(alex_net/batch_normalization/moving_mean2batch_norm1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE,alex_net/batch_normalization/moving_variance6batch_norm1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
,0
-1
.2
/3*

,0
-1*
* 

Õnon_trainable_variables
Ölayers
×metrics
 Ølayer_regularization_losses
Ùlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Únon_trainable_variables
Ûlayers
Ümetrics
 Ýlayer_regularization_losses
Þlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses* 
* 
* 
YS
VARIABLE_VALUEalex_net/conv2d_1/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEalex_net/conv2d_1/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE*

<0
=1*

<0
=1*


Á0* 

ßnon_trainable_variables
àlayers
ámetrics
 âlayer_regularization_losses
ãlayer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses*
* 
* 
* 
jd
VARIABLE_VALUE$alex_net/batch_normalization_1/gamma,batch_norm2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE#alex_net/batch_normalization_1/beta+batch_norm2/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*alex_net/batch_normalization_1/moving_mean2batch_norm2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE.alex_net/batch_normalization_1/moving_variance6batch_norm2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
E0
F1
G2
H3*

E0
F1*
* 

änon_trainable_variables
ålayers
æmetrics
 çlayer_regularization_losses
èlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

énon_trainable_variables
êlayers
ëmetrics
 ìlayer_regularization_losses
ílayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses* 
* 
* 
YS
VARIABLE_VALUEalex_net/conv2d_2/kernel'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEalex_net/conv2d_2/bias%conv3/bias/.ATTRIBUTES/VARIABLE_VALUE*

U0
V1*

U0
V1*


Â0* 

înon_trainable_variables
ïlayers
ðmetrics
 ñlayer_regularization_losses
òlayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses*
* 
* 
* 
jd
VARIABLE_VALUE$alex_net/batch_normalization_2/gamma,batch_norm3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE#alex_net/batch_normalization_2/beta+batch_norm3/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*alex_net/batch_normalization_2/moving_mean2batch_norm3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE.alex_net/batch_normalization_2/moving_variance6batch_norm3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
^0
_1
`2
a3*

^0
_1*
* 

ónon_trainable_variables
ôlayers
õmetrics
 ölayer_regularization_losses
÷layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*
* 
* 
YS
VARIABLE_VALUEalex_net/conv2d_3/kernel'conv4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEalex_net/conv2d_3/bias%conv4/bias/.ATTRIBUTES/VARIABLE_VALUE*

h0
i1*

h0
i1*


Ã0* 

ønon_trainable_variables
ùlayers
úmetrics
 ûlayer_regularization_losses
ülayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses*
* 
* 
* 
jd
VARIABLE_VALUE$alex_net/batch_normalization_3/gamma,batch_norm4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE#alex_net/batch_normalization_3/beta+batch_norm4/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*alex_net/batch_normalization_3/moving_mean2batch_norm4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE.alex_net/batch_normalization_3/moving_variance6batch_norm4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
q0
r1
s2
t3*

q0
r1*
* 

ýnon_trainable_variables
þlayers
ÿmetrics
 layer_regularization_losses
layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses*
* 
* 
YS
VARIABLE_VALUEalex_net/conv2d_4/kernel'conv5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEalex_net/conv2d_4/bias%conv5/bias/.ATTRIBUTES/VARIABLE_VALUE*

{0
|1*

{0
|1*


Ä0* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
}	variables
~trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
jd
VARIABLE_VALUE$alex_net/batch_normalization_4/gamma,batch_norm5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE#alex_net/batch_normalization_4/beta+batch_norm5/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*alex_net/batch_normalization_4/moving_mean2batch_norm5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE.alex_net/batch_normalization_4/moving_variance6batch_norm5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
0
1
2
3*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
WQ
VARIABLE_VALUEalex_net/dense/kernel(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEalex_net/dense/bias&dense1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
+¡&call_and_return_all_conditional_losses
'¡"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¢	variables
£trainable_variables
¤regularization_losses
§__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses* 
* 
* 
* 
YS
VARIABLE_VALUEalex_net/dense_1/kernel(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEalex_net/dense_1/bias&dense2/bias/.ATTRIBUTES/VARIABLE_VALUE*

©0
ª1*

©0
ª1*
* 

 non_trainable_variables
¡layers
¢metrics
 £layer_regularization_losses
¤layer_metrics
«	variables
¬trainable_variables
­regularization_losses
¯__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

¥non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
±	variables
²trainable_variables
³regularization_losses
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses* 
* 
* 
* 
]W
VARIABLE_VALUEalex_net/dense_2/kernel,classifier/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEalex_net/dense_2/bias*classifier/bias/.ATTRIBUTES/VARIABLE_VALUE*

¸0
¹1*

¸0
¹1*
* 

ªnon_trainable_variables
«layers
¬metrics
 ­layer_regularization_losses
®layer_metrics
º	variables
»trainable_variables
¼regularization_losses
¾__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
L
.0
/1
G2
H3
`4
a5
s6
t7
8
9*

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


À0* 
* 

.0
/1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


Á0* 
* 

G0
H1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


Â0* 
* 

`0
a1*
* 
* 
* 
* 
* 
* 
* 


Ã0* 
* 

s0
t1*
* 
* 
* 
* 
* 
* 
* 


Ä0* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

serving_default_input_1Placeholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿãã
º
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1alex_net/conv2d/kernelalex_net/conv2d/bias"alex_net/batch_normalization/gamma!alex_net/batch_normalization/beta(alex_net/batch_normalization/moving_mean,alex_net/batch_normalization/moving_variancealex_net/conv2d_1/kernelalex_net/conv2d_1/bias$alex_net/batch_normalization_1/gamma#alex_net/batch_normalization_1/beta*alex_net/batch_normalization_1/moving_mean.alex_net/batch_normalization_1/moving_variancealex_net/conv2d_2/kernelalex_net/conv2d_2/bias$alex_net/batch_normalization_2/gamma#alex_net/batch_normalization_2/beta*alex_net/batch_normalization_2/moving_mean.alex_net/batch_normalization_2/moving_variancealex_net/conv2d_3/kernelalex_net/conv2d_3/bias$alex_net/batch_normalization_3/gamma#alex_net/batch_normalization_3/beta*alex_net/batch_normalization_3/moving_mean.alex_net/batch_normalization_3/moving_variancealex_net/conv2d_4/kernelalex_net/conv2d_4/bias$alex_net/batch_normalization_4/gamma#alex_net/batch_normalization_4/beta*alex_net/batch_normalization_4/moving_mean.alex_net/batch_normalization_4/moving_variancealex_net/dense/kernelalex_net/dense/biasalex_net/dense_1/kernelalex_net/dense_1/biasalex_net/dense_2/kernelalex_net/dense_2/bias*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_161572
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ï
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*alex_net/conv2d/kernel/Read/ReadVariableOp(alex_net/conv2d/bias/Read/ReadVariableOp6alex_net/batch_normalization/gamma/Read/ReadVariableOp5alex_net/batch_normalization/beta/Read/ReadVariableOp<alex_net/batch_normalization/moving_mean/Read/ReadVariableOp@alex_net/batch_normalization/moving_variance/Read/ReadVariableOp,alex_net/conv2d_1/kernel/Read/ReadVariableOp*alex_net/conv2d_1/bias/Read/ReadVariableOp8alex_net/batch_normalization_1/gamma/Read/ReadVariableOp7alex_net/batch_normalization_1/beta/Read/ReadVariableOp>alex_net/batch_normalization_1/moving_mean/Read/ReadVariableOpBalex_net/batch_normalization_1/moving_variance/Read/ReadVariableOp,alex_net/conv2d_2/kernel/Read/ReadVariableOp*alex_net/conv2d_2/bias/Read/ReadVariableOp8alex_net/batch_normalization_2/gamma/Read/ReadVariableOp7alex_net/batch_normalization_2/beta/Read/ReadVariableOp>alex_net/batch_normalization_2/moving_mean/Read/ReadVariableOpBalex_net/batch_normalization_2/moving_variance/Read/ReadVariableOp,alex_net/conv2d_3/kernel/Read/ReadVariableOp*alex_net/conv2d_3/bias/Read/ReadVariableOp8alex_net/batch_normalization_3/gamma/Read/ReadVariableOp7alex_net/batch_normalization_3/beta/Read/ReadVariableOp>alex_net/batch_normalization_3/moving_mean/Read/ReadVariableOpBalex_net/batch_normalization_3/moving_variance/Read/ReadVariableOp,alex_net/conv2d_4/kernel/Read/ReadVariableOp*alex_net/conv2d_4/bias/Read/ReadVariableOp8alex_net/batch_normalization_4/gamma/Read/ReadVariableOp7alex_net/batch_normalization_4/beta/Read/ReadVariableOp>alex_net/batch_normalization_4/moving_mean/Read/ReadVariableOpBalex_net/batch_normalization_4/moving_variance/Read/ReadVariableOp)alex_net/dense/kernel/Read/ReadVariableOp'alex_net/dense/bias/Read/ReadVariableOp+alex_net/dense_1/kernel/Read/ReadVariableOp)alex_net/dense_1/bias/Read/ReadVariableOp+alex_net/dense_2/kernel/Read/ReadVariableOp)alex_net/dense_2/bias/Read/ReadVariableOpConst*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_162396
ú
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamealex_net/conv2d/kernelalex_net/conv2d/bias"alex_net/batch_normalization/gamma!alex_net/batch_normalization/beta(alex_net/batch_normalization/moving_mean,alex_net/batch_normalization/moving_variancealex_net/conv2d_1/kernelalex_net/conv2d_1/bias$alex_net/batch_normalization_1/gamma#alex_net/batch_normalization_1/beta*alex_net/batch_normalization_1/moving_mean.alex_net/batch_normalization_1/moving_variancealex_net/conv2d_2/kernelalex_net/conv2d_2/bias$alex_net/batch_normalization_2/gamma#alex_net/batch_normalization_2/beta*alex_net/batch_normalization_2/moving_mean.alex_net/batch_normalization_2/moving_variancealex_net/conv2d_3/kernelalex_net/conv2d_3/bias$alex_net/batch_normalization_3/gamma#alex_net/batch_normalization_3/beta*alex_net/batch_normalization_3/moving_mean.alex_net/batch_normalization_3/moving_variancealex_net/conv2d_4/kernelalex_net/conv2d_4/bias$alex_net/batch_normalization_4/gamma#alex_net/batch_normalization_4/beta*alex_net/batch_normalization_4/moving_mean.alex_net/batch_normalization_4/moving_variancealex_net/dense/kernelalex_net/dense/biasalex_net/dense_1/kernelalex_net/dense_1/biasalex_net/dense_2/kernelalex_net/dense_2/bias*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_162514¿
±
á
)__inference_alex_net_layer_call_fn_161137

inputs!
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
	unknown_3:`
	unknown_4:`$
	unknown_5:`
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	H 

unknown_30: 

unknown_31:  

unknown_32: 

unknown_33: 


unknown_34:

identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*<
_read_only_resource_inputs
	
 !"#$*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_alex_net_layer_call_and_return_conditional_losses_160549o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:ÿÿÿÿÿÿÿÿÿãã: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã
 
_user_specified_nameinputs
÷
a
E__inference_rescaling_layer_call_and_return_conditional_losses_159862

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    _
mulMulinputsCast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿããd
addAddV2mul:z:0Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿããY
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿãã:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã
 
_user_specified_nameinputs
Ü
 
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_161765

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¿
D__inference_alex_net_layer_call_and_return_conditional_losses_160131

inputs'
conv2d_159882:`
conv2d_159884:`(
batch_normalization_159887:`(
batch_normalization_159889:`(
batch_normalization_159891:`(
batch_normalization_159893:`*
conv2d_1_159915:`
conv2d_1_159917:	+
batch_normalization_1_159920:	+
batch_normalization_1_159922:	+
batch_normalization_1_159924:	+
batch_normalization_1_159926:	+
conv2d_2_159948:
conv2d_2_159950:	+
batch_normalization_2_159953:	+
batch_normalization_2_159955:	+
batch_normalization_2_159957:	+
batch_normalization_2_159959:	+
conv2d_3_159980:
conv2d_3_159982:	+
batch_normalization_3_159985:	+
batch_normalization_3_159987:	+
batch_normalization_3_159989:	+
batch_normalization_3_159991:	+
conv2d_4_160012:
conv2d_4_160014:	+
batch_normalization_4_160017:	+
batch_normalization_4_160019:	+
batch_normalization_4_160021:	+
batch_normalization_4_160023:	
dense_160047:	H 
dense_160049:  
dense_1_160071:  
dense_1_160073:  
dense_2_160095: 

dense_2_160097:

identity¢8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp¢:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp¢:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp¢:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp¢:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCallÇ
rescaling/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_rescaling_layer_call_and_return_conditional_losses_159862
conv2d/StatefulPartitionedCallStatefulPartitionedCall"rescaling/PartitionedCall:output:0conv2d_159882conv2d_159884*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ77`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_159881
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_159887batch_normalization_159889batch_normalization_159891batch_normalization_159893*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ77`*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_159513û
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_159564
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_159915conv2d_1_159917*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_159914
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_159920batch_normalization_1_159922batch_normalization_1_159924batch_normalization_1_159926*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_159589
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_159640
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_159948conv2d_2_159950*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_159947
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_159953batch_normalization_2_159955batch_normalization_2_159957batch_normalization_2_159959*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_159665¬
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_3_159980conv2d_3_159982*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_159979
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_159985batch_normalization_3_159987batch_normalization_3_159989batch_normalization_3_159991*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_159729¬
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_4_160012conv2d_4_160014*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_160011
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_160017batch_normalization_4_160019batch_normalization_4_160021batch_normalization_4_160023*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_159793
max_pooling2d_2/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_159844Ü
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿH* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_160033
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_160047dense_160049*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_160046Ù
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_160057
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_160071dense_1_160073*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_160070ß
dropout_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_160081
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_160095dense_2_160097*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_160094
8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_159882*&
_output_shapes
:`*
dtype0¦
)alex_net/conv2d/kernel/Regularizer/SquareSquare@alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:`
(alex_net/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             °
&alex_net/conv2d/kernel/Regularizer/SumSum-alex_net/conv2d/kernel/Regularizer/Square:y:01alex_net/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: m
(alex_net/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;²
&alex_net/conv2d/kernel/Regularizer/mulMul1alex_net/conv2d/kernel/Regularizer/mul/x:output:0/alex_net/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_159915*'
_output_shapes
:`*
dtype0«
+alex_net/conv2d_1/kernel/Regularizer/SquareSquareBalex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:`
*alex_net/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_1/kernel/Regularizer/SumSum/alex_net/conv2d_1/kernel/Regularizer/Square:y:03alex_net/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_1/kernel/Regularizer/mulMul3alex_net/conv2d_1/kernel/Regularizer/mul/x:output:01alex_net/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_159948*(
_output_shapes
:*
dtype0¬
+alex_net/conv2d_2/kernel/Regularizer/SquareSquareBalex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:
*alex_net/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_2/kernel/Regularizer/SumSum/alex_net/conv2d_2/kernel/Regularizer/Square:y:03alex_net/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_2/kernel/Regularizer/mulMul3alex_net/conv2d_2/kernel/Regularizer/mul/x:output:01alex_net/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_159980*(
_output_shapes
:*
dtype0¬
+alex_net/conv2d_3/kernel/Regularizer/SquareSquareBalex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:
*alex_net/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_3/kernel/Regularizer/SumSum/alex_net/conv2d_3/kernel/Regularizer/Square:y:03alex_net/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_3/kernel/Regularizer/mulMul3alex_net/conv2d_3/kernel/Regularizer/mul/x:output:01alex_net/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4_160012*(
_output_shapes
:*
dtype0¬
+alex_net/conv2d_4/kernel/Regularizer/SquareSquareBalex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:
*alex_net/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_4/kernel/Regularizer/SumSum/alex_net/conv2d_4/kernel/Regularizer/Square:y:03alex_net/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_4/kernel/Regularizer/mulMul3alex_net/conv2d_4/kernel/Regularizer/mul/x:output:01alex_net/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ô
NoOpNoOp9^alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp;^alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp;^alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp;^alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp;^alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:ÿÿÿÿÿÿÿÿÿãã: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2t
8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp2x
:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2x
:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2x
:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2x
:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã
 
_user_specified_nameinputs
ÕÁ
Ý#
!__inference__wrapped_model_159491
input_1H
.alex_net_conv2d_conv2d_readvariableop_resource:`=
/alex_net_conv2d_biasadd_readvariableop_resource:`B
4alex_net_batch_normalization_readvariableop_resource:`D
6alex_net_batch_normalization_readvariableop_1_resource:`S
Ealex_net_batch_normalization_fusedbatchnormv3_readvariableop_resource:`U
Galex_net_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:`K
0alex_net_conv2d_1_conv2d_readvariableop_resource:`@
1alex_net_conv2d_1_biasadd_readvariableop_resource:	E
6alex_net_batch_normalization_1_readvariableop_resource:	G
8alex_net_batch_normalization_1_readvariableop_1_resource:	V
Galex_net_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	X
Ialex_net_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	L
0alex_net_conv2d_2_conv2d_readvariableop_resource:@
1alex_net_conv2d_2_biasadd_readvariableop_resource:	E
6alex_net_batch_normalization_2_readvariableop_resource:	G
8alex_net_batch_normalization_2_readvariableop_1_resource:	V
Galex_net_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	X
Ialex_net_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	L
0alex_net_conv2d_3_conv2d_readvariableop_resource:@
1alex_net_conv2d_3_biasadd_readvariableop_resource:	E
6alex_net_batch_normalization_3_readvariableop_resource:	G
8alex_net_batch_normalization_3_readvariableop_1_resource:	V
Galex_net_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	X
Ialex_net_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	L
0alex_net_conv2d_4_conv2d_readvariableop_resource:@
1alex_net_conv2d_4_biasadd_readvariableop_resource:	E
6alex_net_batch_normalization_4_readvariableop_resource:	G
8alex_net_batch_normalization_4_readvariableop_1_resource:	V
Galex_net_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	X
Ialex_net_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	@
-alex_net_dense_matmul_readvariableop_resource:	H <
.alex_net_dense_biasadd_readvariableop_resource: A
/alex_net_dense_1_matmul_readvariableop_resource:  >
0alex_net_dense_1_biasadd_readvariableop_resource: A
/alex_net_dense_2_matmul_readvariableop_resource: 
>
0alex_net_dense_2_biasadd_readvariableop_resource:

identity¢<alex_net/batch_normalization/FusedBatchNormV3/ReadVariableOp¢>alex_net/batch_normalization/FusedBatchNormV3/ReadVariableOp_1¢+alex_net/batch_normalization/ReadVariableOp¢-alex_net/batch_normalization/ReadVariableOp_1¢>alex_net/batch_normalization_1/FusedBatchNormV3/ReadVariableOp¢@alex_net/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1¢-alex_net/batch_normalization_1/ReadVariableOp¢/alex_net/batch_normalization_1/ReadVariableOp_1¢>alex_net/batch_normalization_2/FusedBatchNormV3/ReadVariableOp¢@alex_net/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¢-alex_net/batch_normalization_2/ReadVariableOp¢/alex_net/batch_normalization_2/ReadVariableOp_1¢>alex_net/batch_normalization_3/FusedBatchNormV3/ReadVariableOp¢@alex_net/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1¢-alex_net/batch_normalization_3/ReadVariableOp¢/alex_net/batch_normalization_3/ReadVariableOp_1¢>alex_net/batch_normalization_4/FusedBatchNormV3/ReadVariableOp¢@alex_net/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1¢-alex_net/batch_normalization_4/ReadVariableOp¢/alex_net/batch_normalization_4/ReadVariableOp_1¢&alex_net/conv2d/BiasAdd/ReadVariableOp¢%alex_net/conv2d/Conv2D/ReadVariableOp¢(alex_net/conv2d_1/BiasAdd/ReadVariableOp¢'alex_net/conv2d_1/Conv2D/ReadVariableOp¢(alex_net/conv2d_2/BiasAdd/ReadVariableOp¢'alex_net/conv2d_2/Conv2D/ReadVariableOp¢(alex_net/conv2d_3/BiasAdd/ReadVariableOp¢'alex_net/conv2d_3/Conv2D/ReadVariableOp¢(alex_net/conv2d_4/BiasAdd/ReadVariableOp¢'alex_net/conv2d_4/Conv2D/ReadVariableOp¢%alex_net/dense/BiasAdd/ReadVariableOp¢$alex_net/dense/MatMul/ReadVariableOp¢'alex_net/dense_1/BiasAdd/ReadVariableOp¢&alex_net/dense_1/MatMul/ReadVariableOp¢'alex_net/dense_2/BiasAdd/ReadVariableOp¢&alex_net/dense_2/MatMul/ReadVariableOp^
alex_net/rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;`
alex_net/rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
alex_net/rescaling/mulMulinput_1"alex_net/rescaling/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã
alex_net/rescaling/addAddV2alex_net/rescaling/mul:z:0$alex_net/rescaling/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã
%alex_net/conv2d/Conv2D/ReadVariableOpReadVariableOp.alex_net_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:`*
dtype0Î
alex_net/conv2d/Conv2DConv2Dalex_net/rescaling/add:z:0-alex_net/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ77`*
paddingVALID*
strides

&alex_net/conv2d/BiasAdd/ReadVariableOpReadVariableOp/alex_net_conv2d_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0­
alex_net/conv2d/BiasAddBiasAddalex_net/conv2d/Conv2D:output:0.alex_net/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ77`x
alex_net/conv2d/ReluRelu alex_net/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ77`
+alex_net/batch_normalization/ReadVariableOpReadVariableOp4alex_net_batch_normalization_readvariableop_resource*
_output_shapes
:`*
dtype0 
-alex_net/batch_normalization/ReadVariableOp_1ReadVariableOp6alex_net_batch_normalization_readvariableop_1_resource*
_output_shapes
:`*
dtype0¾
<alex_net/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpEalex_net_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0Â
>alex_net/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGalex_net_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0ã
-alex_net/batch_normalization/FusedBatchNormV3FusedBatchNormV3"alex_net/conv2d/Relu:activations:03alex_net/batch_normalization/ReadVariableOp:value:05alex_net/batch_normalization/ReadVariableOp_1:value:0Dalex_net/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Falex_net/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ77`:`:`:`:`:*
epsilon%o:*
is_training( É
alex_net/max_pooling2d/MaxPoolMaxPool1alex_net/batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
ksize
*
paddingVALID*
strides
¡
'alex_net/conv2d_1/Conv2D/ReadVariableOpReadVariableOp0alex_net_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:`*
dtype0ß
alex_net/conv2d_1/Conv2DConv2D'alex_net/max_pooling2d/MaxPool:output:0/alex_net/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

(alex_net/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp1alex_net_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0´
alex_net/conv2d_1/BiasAddBiasAdd!alex_net/conv2d_1/Conv2D:output:00alex_net/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
alex_net/conv2d_1/ReluRelu"alex_net/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
-alex_net/batch_normalization_1/ReadVariableOpReadVariableOp6alex_net_batch_normalization_1_readvariableop_resource*
_output_shapes	
:*
dtype0¥
/alex_net/batch_normalization_1/ReadVariableOp_1ReadVariableOp8alex_net_batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ã
>alex_net/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpGalex_net_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Ç
@alex_net/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIalex_net_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0ô
/alex_net/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3$alex_net/conv2d_1/Relu:activations:05alex_net/batch_normalization_1/ReadVariableOp:value:07alex_net/batch_normalization_1/ReadVariableOp_1:value:0Falex_net/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Halex_net/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( Î
 alex_net/max_pooling2d_1/MaxPoolMaxPool3alex_net/batch_normalization_1/FusedBatchNormV3:y:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¢
'alex_net/conv2d_2/Conv2D/ReadVariableOpReadVariableOp0alex_net_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0á
alex_net/conv2d_2/Conv2DConv2D)alex_net/max_pooling2d_1/MaxPool:output:0/alex_net/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

(alex_net/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp1alex_net_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0´
alex_net/conv2d_2/BiasAddBiasAdd!alex_net/conv2d_2/Conv2D:output:00alex_net/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
alex_net/conv2d_2/ReluRelu"alex_net/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
-alex_net/batch_normalization_2/ReadVariableOpReadVariableOp6alex_net_batch_normalization_2_readvariableop_resource*
_output_shapes	
:*
dtype0¥
/alex_net/batch_normalization_2/ReadVariableOp_1ReadVariableOp8alex_net_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ã
>alex_net/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpGalex_net_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Ç
@alex_net/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIalex_net_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0ô
/alex_net/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3$alex_net/conv2d_2/Relu:activations:05alex_net/batch_normalization_2/ReadVariableOp:value:07alex_net/batch_normalization_2/ReadVariableOp_1:value:0Falex_net/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Halex_net/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ¢
'alex_net/conv2d_3/Conv2D/ReadVariableOpReadVariableOp0alex_net_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ë
alex_net/conv2d_3/Conv2DConv2D3alex_net/batch_normalization_2/FusedBatchNormV3:y:0/alex_net/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

(alex_net/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp1alex_net_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0´
alex_net/conv2d_3/BiasAddBiasAdd!alex_net/conv2d_3/Conv2D:output:00alex_net/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
alex_net/conv2d_3/ReluRelu"alex_net/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
-alex_net/batch_normalization_3/ReadVariableOpReadVariableOp6alex_net_batch_normalization_3_readvariableop_resource*
_output_shapes	
:*
dtype0¥
/alex_net/batch_normalization_3/ReadVariableOp_1ReadVariableOp8alex_net_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ã
>alex_net/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpGalex_net_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Ç
@alex_net/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIalex_net_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0ô
/alex_net/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3$alex_net/conv2d_3/Relu:activations:05alex_net/batch_normalization_3/ReadVariableOp:value:07alex_net/batch_normalization_3/ReadVariableOp_1:value:0Falex_net/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Halex_net/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ¢
'alex_net/conv2d_4/Conv2D/ReadVariableOpReadVariableOp0alex_net_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ë
alex_net/conv2d_4/Conv2DConv2D3alex_net/batch_normalization_3/FusedBatchNormV3:y:0/alex_net/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

(alex_net/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp1alex_net_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0´
alex_net/conv2d_4/BiasAddBiasAdd!alex_net/conv2d_4/Conv2D:output:00alex_net/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
alex_net/conv2d_4/ReluRelu"alex_net/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
-alex_net/batch_normalization_4/ReadVariableOpReadVariableOp6alex_net_batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype0¥
/alex_net/batch_normalization_4/ReadVariableOp_1ReadVariableOp8alex_net_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ã
>alex_net/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpGalex_net_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Ç
@alex_net/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIalex_net_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0ô
/alex_net/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3$alex_net/conv2d_4/Relu:activations:05alex_net/batch_normalization_4/ReadVariableOp:value:07alex_net/batch_normalization_4/ReadVariableOp_1:value:0Falex_net/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Halex_net/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( Î
 alex_net/max_pooling2d_2/MaxPoolMaxPool3alex_net/batch_normalization_4/FusedBatchNormV3:y:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
g
alex_net/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ $  ¢
alex_net/flatten/ReshapeReshape)alex_net/max_pooling2d_2/MaxPool:output:0alex_net/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
$alex_net/dense/MatMul/ReadVariableOpReadVariableOp-alex_net_dense_matmul_readvariableop_resource*
_output_shapes
:	H *
dtype0¢
alex_net/dense/MatMulMatMul!alex_net/flatten/Reshape:output:0,alex_net/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%alex_net/dense/BiasAdd/ReadVariableOpReadVariableOp.alex_net_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0£
alex_net/dense/BiasAddBiasAddalex_net/dense/MatMul:product:0-alex_net/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
alex_net/dense/ReluRelualex_net/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
alex_net/dropout/IdentityIdentity!alex_net/dense/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
&alex_net/dense_1/MatMul/ReadVariableOpReadVariableOp/alex_net_dense_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0§
alex_net/dense_1/MatMulMatMul"alex_net/dropout/Identity:output:0.alex_net/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
'alex_net/dense_1/BiasAdd/ReadVariableOpReadVariableOp0alex_net_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0©
alex_net/dense_1/BiasAddBiasAdd!alex_net/dense_1/MatMul:product:0/alex_net/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
alex_net/dense_1/ReluRelu!alex_net/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
alex_net/dropout_1/IdentityIdentity#alex_net/dense_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
&alex_net/dense_2/MatMul/ReadVariableOpReadVariableOp/alex_net_dense_2_matmul_readvariableop_resource*
_output_shapes

: 
*
dtype0©
alex_net/dense_2/MatMulMatMul$alex_net/dropout_1/Identity:output:0.alex_net/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

'alex_net/dense_2/BiasAdd/ReadVariableOpReadVariableOp0alex_net_dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0©
alex_net/dense_2/BiasAddBiasAdd!alex_net/dense_2/MatMul:product:0/alex_net/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
x
alex_net/dense_2/SoftmaxSoftmax!alex_net/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
IdentityIdentity"alex_net/dense_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ö
NoOpNoOp=^alex_net/batch_normalization/FusedBatchNormV3/ReadVariableOp?^alex_net/batch_normalization/FusedBatchNormV3/ReadVariableOp_1,^alex_net/batch_normalization/ReadVariableOp.^alex_net/batch_normalization/ReadVariableOp_1?^alex_net/batch_normalization_1/FusedBatchNormV3/ReadVariableOpA^alex_net/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1.^alex_net/batch_normalization_1/ReadVariableOp0^alex_net/batch_normalization_1/ReadVariableOp_1?^alex_net/batch_normalization_2/FusedBatchNormV3/ReadVariableOpA^alex_net/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1.^alex_net/batch_normalization_2/ReadVariableOp0^alex_net/batch_normalization_2/ReadVariableOp_1?^alex_net/batch_normalization_3/FusedBatchNormV3/ReadVariableOpA^alex_net/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1.^alex_net/batch_normalization_3/ReadVariableOp0^alex_net/batch_normalization_3/ReadVariableOp_1?^alex_net/batch_normalization_4/FusedBatchNormV3/ReadVariableOpA^alex_net/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1.^alex_net/batch_normalization_4/ReadVariableOp0^alex_net/batch_normalization_4/ReadVariableOp_1'^alex_net/conv2d/BiasAdd/ReadVariableOp&^alex_net/conv2d/Conv2D/ReadVariableOp)^alex_net/conv2d_1/BiasAdd/ReadVariableOp(^alex_net/conv2d_1/Conv2D/ReadVariableOp)^alex_net/conv2d_2/BiasAdd/ReadVariableOp(^alex_net/conv2d_2/Conv2D/ReadVariableOp)^alex_net/conv2d_3/BiasAdd/ReadVariableOp(^alex_net/conv2d_3/Conv2D/ReadVariableOp)^alex_net/conv2d_4/BiasAdd/ReadVariableOp(^alex_net/conv2d_4/Conv2D/ReadVariableOp&^alex_net/dense/BiasAdd/ReadVariableOp%^alex_net/dense/MatMul/ReadVariableOp(^alex_net/dense_1/BiasAdd/ReadVariableOp'^alex_net/dense_1/MatMul/ReadVariableOp(^alex_net/dense_2/BiasAdd/ReadVariableOp'^alex_net/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:ÿÿÿÿÿÿÿÿÿãã: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2|
<alex_net/batch_normalization/FusedBatchNormV3/ReadVariableOp<alex_net/batch_normalization/FusedBatchNormV3/ReadVariableOp2
>alex_net/batch_normalization/FusedBatchNormV3/ReadVariableOp_1>alex_net/batch_normalization/FusedBatchNormV3/ReadVariableOp_12Z
+alex_net/batch_normalization/ReadVariableOp+alex_net/batch_normalization/ReadVariableOp2^
-alex_net/batch_normalization/ReadVariableOp_1-alex_net/batch_normalization/ReadVariableOp_12
>alex_net/batch_normalization_1/FusedBatchNormV3/ReadVariableOp>alex_net/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2
@alex_net/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1@alex_net/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12^
-alex_net/batch_normalization_1/ReadVariableOp-alex_net/batch_normalization_1/ReadVariableOp2b
/alex_net/batch_normalization_1/ReadVariableOp_1/alex_net/batch_normalization_1/ReadVariableOp_12
>alex_net/batch_normalization_2/FusedBatchNormV3/ReadVariableOp>alex_net/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2
@alex_net/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1@alex_net/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12^
-alex_net/batch_normalization_2/ReadVariableOp-alex_net/batch_normalization_2/ReadVariableOp2b
/alex_net/batch_normalization_2/ReadVariableOp_1/alex_net/batch_normalization_2/ReadVariableOp_12
>alex_net/batch_normalization_3/FusedBatchNormV3/ReadVariableOp>alex_net/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2
@alex_net/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1@alex_net/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12^
-alex_net/batch_normalization_3/ReadVariableOp-alex_net/batch_normalization_3/ReadVariableOp2b
/alex_net/batch_normalization_3/ReadVariableOp_1/alex_net/batch_normalization_3/ReadVariableOp_12
>alex_net/batch_normalization_4/FusedBatchNormV3/ReadVariableOp>alex_net/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2
@alex_net/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1@alex_net/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12^
-alex_net/batch_normalization_4/ReadVariableOp-alex_net/batch_normalization_4/ReadVariableOp2b
/alex_net/batch_normalization_4/ReadVariableOp_1/alex_net/batch_normalization_4/ReadVariableOp_12P
&alex_net/conv2d/BiasAdd/ReadVariableOp&alex_net/conv2d/BiasAdd/ReadVariableOp2N
%alex_net/conv2d/Conv2D/ReadVariableOp%alex_net/conv2d/Conv2D/ReadVariableOp2T
(alex_net/conv2d_1/BiasAdd/ReadVariableOp(alex_net/conv2d_1/BiasAdd/ReadVariableOp2R
'alex_net/conv2d_1/Conv2D/ReadVariableOp'alex_net/conv2d_1/Conv2D/ReadVariableOp2T
(alex_net/conv2d_2/BiasAdd/ReadVariableOp(alex_net/conv2d_2/BiasAdd/ReadVariableOp2R
'alex_net/conv2d_2/Conv2D/ReadVariableOp'alex_net/conv2d_2/Conv2D/ReadVariableOp2T
(alex_net/conv2d_3/BiasAdd/ReadVariableOp(alex_net/conv2d_3/BiasAdd/ReadVariableOp2R
'alex_net/conv2d_3/Conv2D/ReadVariableOp'alex_net/conv2d_3/Conv2D/ReadVariableOp2T
(alex_net/conv2d_4/BiasAdd/ReadVariableOp(alex_net/conv2d_4/BiasAdd/ReadVariableOp2R
'alex_net/conv2d_4/Conv2D/ReadVariableOp'alex_net/conv2d_4/Conv2D/ReadVariableOp2N
%alex_net/dense/BiasAdd/ReadVariableOp%alex_net/dense/BiasAdd/ReadVariableOp2L
$alex_net/dense/MatMul/ReadVariableOp$alex_net/dense/MatMul/ReadVariableOp2R
'alex_net/dense_1/BiasAdd/ReadVariableOp'alex_net/dense_1/BiasAdd/ReadVariableOp2P
&alex_net/dense_1/MatMul/ReadVariableOp&alex_net/dense_1/MatMul/ReadVariableOp2R
'alex_net/dense_2/BiasAdd/ReadVariableOp'alex_net/dense_2/BiasAdd/ReadVariableOp2P
&alex_net/dense_2/MatMul/ReadVariableOp&alex_net/dense_2/MatMul/ReadVariableOp:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã
!
_user_specified_name	input_1
Ä
½
D__inference_conv2d_3_layer_call_and_return_conditional_losses_159979

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¬
+alex_net/conv2d_3/kernel/Regularizer/SquareSquareBalex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:
*alex_net/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_3/kernel/Regularizer/SumSum/alex_net/conv2d_3/kernel/Regularizer/Square:y:03alex_net/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_3/kernel/Regularizer/mulMul3alex_net/conv2d_3/kernel/Regularizer/mul/x:output:01alex_net/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp;^alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2x
:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô
¡
)__inference_conv2d_3_layer_call_fn_161902

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_159979x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Í
__inference_loss_fn_3_162254_
Calex_net_conv2d_3_kernel_regularizer_square_readvariableop_resource:
identity¢:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOpÈ
:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpCalex_net_conv2d_3_kernel_regularizer_square_readvariableop_resource*(
_output_shapes
:*
dtype0¬
+alex_net/conv2d_3/kernel/Regularizer/SquareSquareBalex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:
*alex_net/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_3/kernel/Regularizer/SumSum/alex_net/conv2d_3/kernel/Regularizer/Square:y:03alex_net/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_3/kernel/Regularizer/mulMul3alex_net/conv2d_3/kernel/Regularizer/mul/x:output:01alex_net/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentity,alex_net/conv2d_3/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp;^alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2x
:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp
	
Õ
6__inference_batch_normalization_3_layer_call_fn_161945

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_159760
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
a
C__inference_dropout_layer_call_and_return_conditional_losses_160057

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
	
Õ
6__inference_batch_normalization_1_layer_call_fn_161734

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_159589
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²
D
(__inference_flatten_layer_call_fn_162090

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿH* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_160033a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿH"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
 
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_161963

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
¾
O__inference_batch_normalization_layer_call_and_return_conditional_losses_159544

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`:`:`:`:`:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
ó	
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_162190

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¾
¼
D__inference_conv2d_1_layer_call_and_return_conditional_losses_159914

inputs9
conv2d_readvariableop_resource:`.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:`*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:`*
dtype0«
+alex_net/conv2d_1/kernel/Regularizer/SquareSquareBalex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:`
*alex_net/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_1/kernel/Regularizer/SumSum/alex_net/conv2d_1/kernel/Regularizer/Square:y:03alex_net/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_1/kernel/Regularizer/mulMul3alex_net/conv2d_1/kernel/Regularizer/mul/x:output:01alex_net/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp;^alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2x
:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
	
Õ
6__inference_batch_normalization_2_layer_call_fn_161838

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_159665
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´
â
)__inference_alex_net_layer_call_fn_160701
input_1!
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
	unknown_3:`
	unknown_4:`$
	unknown_5:`
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	H 

unknown_30: 

unknown_31:  

unknown_32: 

unknown_33: 


unknown_34:

identity¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*<
_read_only_resource_inputs
	
 !"#$*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_alex_net_layer_call_and_return_conditional_losses_160549o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:ÿÿÿÿÿÿÿÿÿãã: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã
!
_user_specified_name	input_1
Ø
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_162178

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ü
 
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_162057

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
Ä
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_159696

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä
½
D__inference_conv2d_3_layer_call_and_return_conditional_losses_161919

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¬
+alex_net/conv2d_3/kernel/Regularizer/SquareSquareBalex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:
*alex_net/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_3/kernel/Regularizer/SumSum/alex_net/conv2d_3/kernel/Regularizer/Square:y:03alex_net/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_3/kernel/Regularizer/mulMul3alex_net/conv2d_3/kernel/Regularizer/mul/x:output:01alex_net/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp;^alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2x
:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_160081

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¾
¼
D__inference_conv2d_1_layer_call_and_return_conditional_losses_161721

inputs9
conv2d_readvariableop_resource:`.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:`*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:`*
dtype0«
+alex_net/conv2d_1/kernel/Regularizer/SquareSquareBalex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:`
*alex_net/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_1/kernel/Regularizer/SumSum/alex_net/conv2d_1/kernel/Regularizer/Square:y:03alex_net/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_1/kernel/Regularizer/mulMul3alex_net/conv2d_1/kernel/Regularizer/mul/x:output:01alex_net/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp;^alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2x
:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs

e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_159564

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
L
0__inference_max_pooling2d_1_layer_call_fn_161788

inputs
identityÜ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_159640
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
Ä
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_159760

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
Ä
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_159824

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ý
$__inference_signature_wrapper_161572
input_1!
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
	unknown_3:`
	unknown_4:`$
	unknown_5:`
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	H 

unknown_30: 

unknown_31:  

unknown_32: 

unknown_33: 


unknown_34:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_159491o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:ÿÿÿÿÿÿÿÿÿãã: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã
!
_user_specified_name	input_1


ô
C__inference_dense_1_layer_call_and_return_conditional_losses_160070

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ü
 
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_161869

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ó
A__inference_dense_layer_call_and_return_conditional_losses_160046

inputs1
matmul_readvariableop_resource:	H -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿH: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs
	
Õ
6__inference_batch_normalization_4_layer_call_fn_162026

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_159793
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç
_
C__inference_flatten_layer_call_and_return_conditional_losses_160033

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ $  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿHY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿH"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
 
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_159665

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ	
b
C__inference_dropout_layer_call_and_return_conditional_losses_160269

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_162085

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ

D__inference_alex_net_layer_call_and_return_conditional_losses_160549

inputs'
conv2d_160427:`
conv2d_160429:`(
batch_normalization_160432:`(
batch_normalization_160434:`(
batch_normalization_160436:`(
batch_normalization_160438:`*
conv2d_1_160442:`
conv2d_1_160444:	+
batch_normalization_1_160447:	+
batch_normalization_1_160449:	+
batch_normalization_1_160451:	+
batch_normalization_1_160453:	+
conv2d_2_160457:
conv2d_2_160459:	+
batch_normalization_2_160462:	+
batch_normalization_2_160464:	+
batch_normalization_2_160466:	+
batch_normalization_2_160468:	+
conv2d_3_160471:
conv2d_3_160473:	+
batch_normalization_3_160476:	+
batch_normalization_3_160478:	+
batch_normalization_3_160480:	+
batch_normalization_3_160482:	+
conv2d_4_160485:
conv2d_4_160487:	+
batch_normalization_4_160490:	+
batch_normalization_4_160492:	+
batch_normalization_4_160494:	+
batch_normalization_4_160496:	
dense_160501:	H 
dense_160503:  
dense_1_160507:  
dense_1_160509:  
dense_2_160513: 

dense_2_160515:

identity¢8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp¢:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp¢:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp¢:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp¢:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCallÇ
rescaling/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_rescaling_layer_call_and_return_conditional_losses_159862
conv2d/StatefulPartitionedCallStatefulPartitionedCall"rescaling/PartitionedCall:output:0conv2d_160427conv2d_160429*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ77`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_159881
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_160432batch_normalization_160434batch_normalization_160436batch_normalization_160438*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ77`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_159544û
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_159564
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_160442conv2d_1_160444*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_159914
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_160447batch_normalization_1_160449batch_normalization_1_160451batch_normalization_1_160453*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_159620
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_159640
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_160457conv2d_2_160459*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_159947
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_160462batch_normalization_2_160464batch_normalization_2_160466batch_normalization_2_160468*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_159696¬
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_3_160471conv2d_3_160473*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_159979
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_160476batch_normalization_3_160478batch_normalization_3_160480batch_normalization_3_160482*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_159760¬
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_4_160485conv2d_4_160487*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_160011
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_160490batch_normalization_4_160492batch_normalization_4_160494batch_normalization_4_160496*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_159824
max_pooling2d_2/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_159844Ü
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿH* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_160033
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_160501dense_160503*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_160046é
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_160269
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_160507dense_1_160509*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_160070
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_160236
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_160513dense_2_160515*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_160094
8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_160427*&
_output_shapes
:`*
dtype0¦
)alex_net/conv2d/kernel/Regularizer/SquareSquare@alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:`
(alex_net/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             °
&alex_net/conv2d/kernel/Regularizer/SumSum-alex_net/conv2d/kernel/Regularizer/Square:y:01alex_net/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: m
(alex_net/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;²
&alex_net/conv2d/kernel/Regularizer/mulMul1alex_net/conv2d/kernel/Regularizer/mul/x:output:0/alex_net/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_160442*'
_output_shapes
:`*
dtype0«
+alex_net/conv2d_1/kernel/Regularizer/SquareSquareBalex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:`
*alex_net/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_1/kernel/Regularizer/SumSum/alex_net/conv2d_1/kernel/Regularizer/Square:y:03alex_net/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_1/kernel/Regularizer/mulMul3alex_net/conv2d_1/kernel/Regularizer/mul/x:output:01alex_net/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_160457*(
_output_shapes
:*
dtype0¬
+alex_net/conv2d_2/kernel/Regularizer/SquareSquareBalex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:
*alex_net/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_2/kernel/Regularizer/SumSum/alex_net/conv2d_2/kernel/Regularizer/Square:y:03alex_net/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_2/kernel/Regularizer/mulMul3alex_net/conv2d_2/kernel/Regularizer/mul/x:output:01alex_net/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_160471*(
_output_shapes
:*
dtype0¬
+alex_net/conv2d_3/kernel/Regularizer/SquareSquareBalex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:
*alex_net/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_3/kernel/Regularizer/SumSum/alex_net/conv2d_3/kernel/Regularizer/Square:y:03alex_net/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_3/kernel/Regularizer/mulMul3alex_net/conv2d_3/kernel/Regularizer/mul/x:output:01alex_net/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4_160485*(
_output_shapes
:*
dtype0¬
+alex_net/conv2d_4/kernel/Regularizer/SquareSquareBalex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:
*alex_net/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_4/kernel/Regularizer/SumSum/alex_net/conv2d_4/kernel/Regularizer/Square:y:03alex_net/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_4/kernel/Regularizer/mulMul3alex_net/conv2d_4/kernel/Regularizer/mul/x:output:01alex_net/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
º
NoOpNoOp9^alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp;^alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp;^alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp;^alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp;^alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:ÿÿÿÿÿÿÿÿÿãã: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2t
8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp2x
:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2x
:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2x
:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2x
:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã
 
_user_specified_nameinputs
¢
À
D__inference_alex_net_layer_call_and_return_conditional_losses_160827
input_1'
conv2d_160705:`
conv2d_160707:`(
batch_normalization_160710:`(
batch_normalization_160712:`(
batch_normalization_160714:`(
batch_normalization_160716:`*
conv2d_1_160720:`
conv2d_1_160722:	+
batch_normalization_1_160725:	+
batch_normalization_1_160727:	+
batch_normalization_1_160729:	+
batch_normalization_1_160731:	+
conv2d_2_160735:
conv2d_2_160737:	+
batch_normalization_2_160740:	+
batch_normalization_2_160742:	+
batch_normalization_2_160744:	+
batch_normalization_2_160746:	+
conv2d_3_160749:
conv2d_3_160751:	+
batch_normalization_3_160754:	+
batch_normalization_3_160756:	+
batch_normalization_3_160758:	+
batch_normalization_3_160760:	+
conv2d_4_160763:
conv2d_4_160765:	+
batch_normalization_4_160768:	+
batch_normalization_4_160770:	+
batch_normalization_4_160772:	+
batch_normalization_4_160774:	
dense_160779:	H 
dense_160781:  
dense_1_160785:  
dense_1_160787:  
dense_2_160791: 

dense_2_160793:

identity¢8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp¢:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp¢:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp¢:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp¢:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCallÈ
rescaling/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_rescaling_layer_call_and_return_conditional_losses_159862
conv2d/StatefulPartitionedCallStatefulPartitionedCall"rescaling/PartitionedCall:output:0conv2d_160705conv2d_160707*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ77`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_159881
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_160710batch_normalization_160712batch_normalization_160714batch_normalization_160716*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ77`*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_159513û
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_159564
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_160720conv2d_1_160722*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_159914
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_160725batch_normalization_1_160727batch_normalization_1_160729batch_normalization_1_160731*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_159589
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_159640
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_160735conv2d_2_160737*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_159947
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_160740batch_normalization_2_160742batch_normalization_2_160744batch_normalization_2_160746*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_159665¬
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_3_160749conv2d_3_160751*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_159979
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_160754batch_normalization_3_160756batch_normalization_3_160758batch_normalization_3_160760*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_159729¬
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_4_160763conv2d_4_160765*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_160011
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_160768batch_normalization_4_160770batch_normalization_4_160772batch_normalization_4_160774*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_159793
max_pooling2d_2/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_159844Ü
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿH* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_160033
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_160779dense_160781*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_160046Ù
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_160057
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_160785dense_1_160787*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_160070ß
dropout_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_160081
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_160791dense_2_160793*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_160094
8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_160705*&
_output_shapes
:`*
dtype0¦
)alex_net/conv2d/kernel/Regularizer/SquareSquare@alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:`
(alex_net/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             °
&alex_net/conv2d/kernel/Regularizer/SumSum-alex_net/conv2d/kernel/Regularizer/Square:y:01alex_net/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: m
(alex_net/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;²
&alex_net/conv2d/kernel/Regularizer/mulMul1alex_net/conv2d/kernel/Regularizer/mul/x:output:0/alex_net/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_160720*'
_output_shapes
:`*
dtype0«
+alex_net/conv2d_1/kernel/Regularizer/SquareSquareBalex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:`
*alex_net/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_1/kernel/Regularizer/SumSum/alex_net/conv2d_1/kernel/Regularizer/Square:y:03alex_net/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_1/kernel/Regularizer/mulMul3alex_net/conv2d_1/kernel/Regularizer/mul/x:output:01alex_net/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_160735*(
_output_shapes
:*
dtype0¬
+alex_net/conv2d_2/kernel/Regularizer/SquareSquareBalex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:
*alex_net/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_2/kernel/Regularizer/SumSum/alex_net/conv2d_2/kernel/Regularizer/Square:y:03alex_net/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_2/kernel/Regularizer/mulMul3alex_net/conv2d_2/kernel/Regularizer/mul/x:output:01alex_net/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_160749*(
_output_shapes
:*
dtype0¬
+alex_net/conv2d_3/kernel/Regularizer/SquareSquareBalex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:
*alex_net/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_3/kernel/Regularizer/SumSum/alex_net/conv2d_3/kernel/Regularizer/Square:y:03alex_net/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_3/kernel/Regularizer/mulMul3alex_net/conv2d_3/kernel/Regularizer/mul/x:output:01alex_net/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4_160763*(
_output_shapes
:*
dtype0¬
+alex_net/conv2d_4/kernel/Regularizer/SquareSquareBalex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:
*alex_net/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_4/kernel/Regularizer/SumSum/alex_net/conv2d_4/kernel/Regularizer/Square:y:03alex_net/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_4/kernel/Regularizer/mulMul3alex_net/conv2d_4/kernel/Regularizer/mul/x:output:01alex_net/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ô
NoOpNoOp9^alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp;^alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp;^alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp;^alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp;^alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:ÿÿÿÿÿÿÿÿÿãã: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2t
8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp2x
:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2x
:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2x
:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2x
:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã
!
_user_specified_name	input_1

¶
B__inference_conv2d_layer_call_and_return_conditional_losses_161617

inputs8
conv2d_readvariableop_resource:`-
biasadd_readvariableop_resource:`
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:`*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ77`*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ77`X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ77`
8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:`*
dtype0¦
)alex_net/conv2d/kernel/Regularizer/SquareSquare@alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:`
(alex_net/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             °
&alex_net/conv2d/kernel/Regularizer/SumSum-alex_net/conv2d/kernel/Regularizer/Square:y:01alex_net/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: m
(alex_net/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;²
&alex_net/conv2d/kernel/Regularizer/mulMul1alex_net/conv2d/kernel/Regularizer/mul/x:output:0/alex_net/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ77`²
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp9^alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿãã: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2t
8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã
 
_user_specified_nameinputs
ê
Ä
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_161783

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·
J
.__inference_max_pooling2d_layer_call_fn_161684

inputs
identityÚ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_159564
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
Ä
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_162075

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
Ä
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_161887

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô
¡
)__inference_conv2d_2_layer_call_fn_161808

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_159947x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


D__inference_alex_net_layer_call_and_return_conditional_losses_160953
input_1'
conv2d_160831:`
conv2d_160833:`(
batch_normalization_160836:`(
batch_normalization_160838:`(
batch_normalization_160840:`(
batch_normalization_160842:`*
conv2d_1_160846:`
conv2d_1_160848:	+
batch_normalization_1_160851:	+
batch_normalization_1_160853:	+
batch_normalization_1_160855:	+
batch_normalization_1_160857:	+
conv2d_2_160861:
conv2d_2_160863:	+
batch_normalization_2_160866:	+
batch_normalization_2_160868:	+
batch_normalization_2_160870:	+
batch_normalization_2_160872:	+
conv2d_3_160875:
conv2d_3_160877:	+
batch_normalization_3_160880:	+
batch_normalization_3_160882:	+
batch_normalization_3_160884:	+
batch_normalization_3_160886:	+
conv2d_4_160889:
conv2d_4_160891:	+
batch_normalization_4_160894:	+
batch_normalization_4_160896:	+
batch_normalization_4_160898:	+
batch_normalization_4_160900:	
dense_160905:	H 
dense_160907:  
dense_1_160911:  
dense_1_160913:  
dense_2_160917: 

dense_2_160919:

identity¢8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp¢:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp¢:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp¢:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp¢:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCallÈ
rescaling/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_rescaling_layer_call_and_return_conditional_losses_159862
conv2d/StatefulPartitionedCallStatefulPartitionedCall"rescaling/PartitionedCall:output:0conv2d_160831conv2d_160833*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ77`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_159881
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_160836batch_normalization_160838batch_normalization_160840batch_normalization_160842*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ77`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_159544û
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_159564
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_160846conv2d_1_160848*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_159914
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_160851batch_normalization_1_160853batch_normalization_1_160855batch_normalization_1_160857*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_159620
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_159640
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_160861conv2d_2_160863*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_159947
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_160866batch_normalization_2_160868batch_normalization_2_160870batch_normalization_2_160872*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_159696¬
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_3_160875conv2d_3_160877*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_159979
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_160880batch_normalization_3_160882batch_normalization_3_160884batch_normalization_3_160886*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_159760¬
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_4_160889conv2d_4_160891*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_160011
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_160894batch_normalization_4_160896batch_normalization_4_160898batch_normalization_4_160900*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_159824
max_pooling2d_2/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_159844Ü
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿH* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_160033
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_160905dense_160907*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_160046é
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_160269
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_160911dense_1_160913*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_160070
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_160236
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_160917dense_2_160919*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_160094
8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_160831*&
_output_shapes
:`*
dtype0¦
)alex_net/conv2d/kernel/Regularizer/SquareSquare@alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:`
(alex_net/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             °
&alex_net/conv2d/kernel/Regularizer/SumSum-alex_net/conv2d/kernel/Regularizer/Square:y:01alex_net/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: m
(alex_net/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;²
&alex_net/conv2d/kernel/Regularizer/mulMul1alex_net/conv2d/kernel/Regularizer/mul/x:output:0/alex_net/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_160846*'
_output_shapes
:`*
dtype0«
+alex_net/conv2d_1/kernel/Regularizer/SquareSquareBalex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:`
*alex_net/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_1/kernel/Regularizer/SumSum/alex_net/conv2d_1/kernel/Regularizer/Square:y:03alex_net/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_1/kernel/Regularizer/mulMul3alex_net/conv2d_1/kernel/Regularizer/mul/x:output:01alex_net/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_160861*(
_output_shapes
:*
dtype0¬
+alex_net/conv2d_2/kernel/Regularizer/SquareSquareBalex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:
*alex_net/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_2/kernel/Regularizer/SumSum/alex_net/conv2d_2/kernel/Regularizer/Square:y:03alex_net/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_2/kernel/Regularizer/mulMul3alex_net/conv2d_2/kernel/Regularizer/mul/x:output:01alex_net/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_160875*(
_output_shapes
:*
dtype0¬
+alex_net/conv2d_3/kernel/Regularizer/SquareSquareBalex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:
*alex_net/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_3/kernel/Regularizer/SumSum/alex_net/conv2d_3/kernel/Regularizer/Square:y:03alex_net/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_3/kernel/Regularizer/mulMul3alex_net/conv2d_3/kernel/Regularizer/mul/x:output:01alex_net/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4_160889*(
_output_shapes
:*
dtype0¬
+alex_net/conv2d_4/kernel/Regularizer/SquareSquareBalex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:
*alex_net/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_4/kernel/Regularizer/SumSum/alex_net/conv2d_4/kernel/Regularizer/Square:y:03alex_net/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_4/kernel/Regularizer/mulMul3alex_net/conv2d_4/kernel/Regularizer/mul/x:output:01alex_net/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
º
NoOpNoOp9^alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp;^alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp;^alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp;^alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp;^alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:ÿÿÿÿÿÿÿÿÿãã: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2t
8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp2x
:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2x
:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2x
:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2x
:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã
!
_user_specified_name	input_1
ê
Ä
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_159620

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Í
__inference_loss_fn_4_162265_
Calex_net_conv2d_4_kernel_regularizer_square_readvariableop_resource:
identity¢:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOpÈ
:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpCalex_net_conv2d_4_kernel_regularizer_square_readvariableop_resource*(
_output_shapes
:*
dtype0¬
+alex_net/conv2d_4/kernel/Regularizer/SquareSquareBalex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:
*alex_net/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_4/kernel/Regularizer/SumSum/alex_net/conv2d_4/kernel/Regularizer/Square:y:03alex_net/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_4/kernel/Regularizer/mulMul3alex_net/conv2d_4/kernel/Regularizer/mul/x:output:01alex_net/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentity,alex_net/conv2d_4/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp;^alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2x
:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp

g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_161793

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
L
0__inference_max_pooling2d_2_layer_call_fn_162080

inputs
identityÜ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_159844
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ	
b
C__inference_dropout_layer_call_and_return_conditional_losses_162143

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ýN

__inference__traced_save_162396
file_prefix5
1savev2_alex_net_conv2d_kernel_read_readvariableop3
/savev2_alex_net_conv2d_bias_read_readvariableopA
=savev2_alex_net_batch_normalization_gamma_read_readvariableop@
<savev2_alex_net_batch_normalization_beta_read_readvariableopG
Csavev2_alex_net_batch_normalization_moving_mean_read_readvariableopK
Gsavev2_alex_net_batch_normalization_moving_variance_read_readvariableop7
3savev2_alex_net_conv2d_1_kernel_read_readvariableop5
1savev2_alex_net_conv2d_1_bias_read_readvariableopC
?savev2_alex_net_batch_normalization_1_gamma_read_readvariableopB
>savev2_alex_net_batch_normalization_1_beta_read_readvariableopI
Esavev2_alex_net_batch_normalization_1_moving_mean_read_readvariableopM
Isavev2_alex_net_batch_normalization_1_moving_variance_read_readvariableop7
3savev2_alex_net_conv2d_2_kernel_read_readvariableop5
1savev2_alex_net_conv2d_2_bias_read_readvariableopC
?savev2_alex_net_batch_normalization_2_gamma_read_readvariableopB
>savev2_alex_net_batch_normalization_2_beta_read_readvariableopI
Esavev2_alex_net_batch_normalization_2_moving_mean_read_readvariableopM
Isavev2_alex_net_batch_normalization_2_moving_variance_read_readvariableop7
3savev2_alex_net_conv2d_3_kernel_read_readvariableop5
1savev2_alex_net_conv2d_3_bias_read_readvariableopC
?savev2_alex_net_batch_normalization_3_gamma_read_readvariableopB
>savev2_alex_net_batch_normalization_3_beta_read_readvariableopI
Esavev2_alex_net_batch_normalization_3_moving_mean_read_readvariableopM
Isavev2_alex_net_batch_normalization_3_moving_variance_read_readvariableop7
3savev2_alex_net_conv2d_4_kernel_read_readvariableop5
1savev2_alex_net_conv2d_4_bias_read_readvariableopC
?savev2_alex_net_batch_normalization_4_gamma_read_readvariableopB
>savev2_alex_net_batch_normalization_4_beta_read_readvariableopI
Esavev2_alex_net_batch_normalization_4_moving_mean_read_readvariableopM
Isavev2_alex_net_batch_normalization_4_moving_variance_read_readvariableop4
0savev2_alex_net_dense_kernel_read_readvariableop2
.savev2_alex_net_dense_bias_read_readvariableop6
2savev2_alex_net_dense_1_kernel_read_readvariableop4
0savev2_alex_net_dense_1_bias_read_readvariableop6
2savev2_alex_net_dense_2_kernel_read_readvariableop4
0savev2_alex_net_dense_2_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: û
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*¤
valueB%B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB,batch_norm1/gamma/.ATTRIBUTES/VARIABLE_VALUEB+batch_norm1/beta/.ATTRIBUTES/VARIABLE_VALUEB2batch_norm1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB6batch_norm1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB,batch_norm2/gamma/.ATTRIBUTES/VARIABLE_VALUEB+batch_norm2/beta/.ATTRIBUTES/VARIABLE_VALUEB2batch_norm2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB6batch_norm2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB,batch_norm3/gamma/.ATTRIBUTES/VARIABLE_VALUEB+batch_norm3/beta/.ATTRIBUTES/VARIABLE_VALUEB2batch_norm3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB6batch_norm3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB'conv4/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv4/bias/.ATTRIBUTES/VARIABLE_VALUEB,batch_norm4/gamma/.ATTRIBUTES/VARIABLE_VALUEB+batch_norm4/beta/.ATTRIBUTES/VARIABLE_VALUEB2batch_norm4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB6batch_norm4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB'conv5/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv5/bias/.ATTRIBUTES/VARIABLE_VALUEB,batch_norm5/gamma/.ATTRIBUTES/VARIABLE_VALUEB+batch_norm5/beta/.ATTRIBUTES/VARIABLE_VALUEB2batch_norm5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB6batch_norm5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense2/bias/.ATTRIBUTES/VARIABLE_VALUEB,classifier/kernel/.ATTRIBUTES/VARIABLE_VALUEB*classifier/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH·
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ý
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_alex_net_conv2d_kernel_read_readvariableop/savev2_alex_net_conv2d_bias_read_readvariableop=savev2_alex_net_batch_normalization_gamma_read_readvariableop<savev2_alex_net_batch_normalization_beta_read_readvariableopCsavev2_alex_net_batch_normalization_moving_mean_read_readvariableopGsavev2_alex_net_batch_normalization_moving_variance_read_readvariableop3savev2_alex_net_conv2d_1_kernel_read_readvariableop1savev2_alex_net_conv2d_1_bias_read_readvariableop?savev2_alex_net_batch_normalization_1_gamma_read_readvariableop>savev2_alex_net_batch_normalization_1_beta_read_readvariableopEsavev2_alex_net_batch_normalization_1_moving_mean_read_readvariableopIsavev2_alex_net_batch_normalization_1_moving_variance_read_readvariableop3savev2_alex_net_conv2d_2_kernel_read_readvariableop1savev2_alex_net_conv2d_2_bias_read_readvariableop?savev2_alex_net_batch_normalization_2_gamma_read_readvariableop>savev2_alex_net_batch_normalization_2_beta_read_readvariableopEsavev2_alex_net_batch_normalization_2_moving_mean_read_readvariableopIsavev2_alex_net_batch_normalization_2_moving_variance_read_readvariableop3savev2_alex_net_conv2d_3_kernel_read_readvariableop1savev2_alex_net_conv2d_3_bias_read_readvariableop?savev2_alex_net_batch_normalization_3_gamma_read_readvariableop>savev2_alex_net_batch_normalization_3_beta_read_readvariableopEsavev2_alex_net_batch_normalization_3_moving_mean_read_readvariableopIsavev2_alex_net_batch_normalization_3_moving_variance_read_readvariableop3savev2_alex_net_conv2d_4_kernel_read_readvariableop1savev2_alex_net_conv2d_4_bias_read_readvariableop?savev2_alex_net_batch_normalization_4_gamma_read_readvariableop>savev2_alex_net_batch_normalization_4_beta_read_readvariableopEsavev2_alex_net_batch_normalization_4_moving_mean_read_readvariableopIsavev2_alex_net_batch_normalization_4_moving_variance_read_readvariableop0savev2_alex_net_dense_kernel_read_readvariableop.savev2_alex_net_dense_bias_read_readvariableop2savev2_alex_net_dense_1_kernel_read_readvariableop0savev2_alex_net_dense_1_bias_read_readvariableop2savev2_alex_net_dense_2_kernel_read_readvariableop0savev2_alex_net_dense_2_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *3
dtypes)
'2%
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Õ
_input_shapesÃ
À: :`:`:`:`:`:`:`::::::::::::::::::::::::	H : :  : : 
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:`: 

_output_shapes
:`: 

_output_shapes
:`: 

_output_shapes
:`: 

_output_shapes
:`: 

_output_shapes
:`:-)
'
_output_shapes
:`:!

_output_shapes	
::!	

_output_shapes	
::!


_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::%!

_output_shapes
:	H :  

_output_shapes
: :$! 

_output_shapes

:  : "

_output_shapes
: :$# 

_output_shapes

: 
: $

_output_shapes
:
:%

_output_shapes
: 
ô
c
*__inference_dropout_1_layer_call_fn_162173

inputs
identity¢StatefulPartitionedCallÃ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_160236o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
í

'__inference_conv2d_layer_call_fn_161600

inputs!
unknown:`
	unknown_0:`
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ77`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_159881w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ77``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿãã: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã
 
_user_specified_nameinputs
÷
a
E__inference_rescaling_layer_call_and_return_conditional_losses_161585

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    _
mulMulinputsCast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿããd
addAddV2mul:z:0Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿããY
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿãã:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã
 
_user_specified_nameinputs
	
Õ
6__inference_batch_normalization_1_layer_call_fn_161747

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_159620
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä
½
D__inference_conv2d_2_layer_call_and_return_conditional_losses_161825

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¬
+alex_net/conv2d_2/kernel/Regularizer/SquareSquareBalex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:
*alex_net/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_2/kernel/Regularizer/SumSum/alex_net/conv2d_2/kernel/Regularizer/Square:y:03alex_net/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_2/kernel/Regularizer/mulMul3alex_net/conv2d_2/kernel/Regularizer/mul/x:output:01alex_net/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp;^alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2x
:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð
a
(__inference_dropout_layer_call_fn_162126

inputs
identity¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_160269o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
×
Ç
__inference_loss_fn_0_162221[
Aalex_net_conv2d_kernel_regularizer_square_readvariableop_resource:`
identity¢8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOpÂ
8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpAalex_net_conv2d_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:`*
dtype0¦
)alex_net/conv2d/kernel/Regularizer/SquareSquare@alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:`
(alex_net/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             °
&alex_net/conv2d/kernel/Regularizer/SumSum-alex_net/conv2d/kernel/Regularizer/Square:y:01alex_net/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: m
(alex_net/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;²
&alex_net/conv2d/kernel/Regularizer/mulMul1alex_net/conv2d/kernel/Regularizer/mul/x:output:0/alex_net/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: h
IdentityIdentity*alex_net/conv2d/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp9^alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2t
8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp
¢
F
*__inference_dropout_1_layer_call_fn_162168

inputs
identity³
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_160081`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ý
²
"__inference__traced_restore_162514
file_prefixA
'assignvariableop_alex_net_conv2d_kernel:`5
'assignvariableop_1_alex_net_conv2d_bias:`C
5assignvariableop_2_alex_net_batch_normalization_gamma:`B
4assignvariableop_3_alex_net_batch_normalization_beta:`I
;assignvariableop_4_alex_net_batch_normalization_moving_mean:`M
?assignvariableop_5_alex_net_batch_normalization_moving_variance:`F
+assignvariableop_6_alex_net_conv2d_1_kernel:`8
)assignvariableop_7_alex_net_conv2d_1_bias:	F
7assignvariableop_8_alex_net_batch_normalization_1_gamma:	E
6assignvariableop_9_alex_net_batch_normalization_1_beta:	M
>assignvariableop_10_alex_net_batch_normalization_1_moving_mean:	Q
Bassignvariableop_11_alex_net_batch_normalization_1_moving_variance:	H
,assignvariableop_12_alex_net_conv2d_2_kernel:9
*assignvariableop_13_alex_net_conv2d_2_bias:	G
8assignvariableop_14_alex_net_batch_normalization_2_gamma:	F
7assignvariableop_15_alex_net_batch_normalization_2_beta:	M
>assignvariableop_16_alex_net_batch_normalization_2_moving_mean:	Q
Bassignvariableop_17_alex_net_batch_normalization_2_moving_variance:	H
,assignvariableop_18_alex_net_conv2d_3_kernel:9
*assignvariableop_19_alex_net_conv2d_3_bias:	G
8assignvariableop_20_alex_net_batch_normalization_3_gamma:	F
7assignvariableop_21_alex_net_batch_normalization_3_beta:	M
>assignvariableop_22_alex_net_batch_normalization_3_moving_mean:	Q
Bassignvariableop_23_alex_net_batch_normalization_3_moving_variance:	H
,assignvariableop_24_alex_net_conv2d_4_kernel:9
*assignvariableop_25_alex_net_conv2d_4_bias:	G
8assignvariableop_26_alex_net_batch_normalization_4_gamma:	F
7assignvariableop_27_alex_net_batch_normalization_4_beta:	M
>assignvariableop_28_alex_net_batch_normalization_4_moving_mean:	Q
Bassignvariableop_29_alex_net_batch_normalization_4_moving_variance:	<
)assignvariableop_30_alex_net_dense_kernel:	H 5
'assignvariableop_31_alex_net_dense_bias: =
+assignvariableop_32_alex_net_dense_1_kernel:  7
)assignvariableop_33_alex_net_dense_1_bias: =
+assignvariableop_34_alex_net_dense_2_kernel: 
7
)assignvariableop_35_alex_net_dense_2_bias:

identity_37¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9þ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*¤
valueB%B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB,batch_norm1/gamma/.ATTRIBUTES/VARIABLE_VALUEB+batch_norm1/beta/.ATTRIBUTES/VARIABLE_VALUEB2batch_norm1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB6batch_norm1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB,batch_norm2/gamma/.ATTRIBUTES/VARIABLE_VALUEB+batch_norm2/beta/.ATTRIBUTES/VARIABLE_VALUEB2batch_norm2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB6batch_norm2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB,batch_norm3/gamma/.ATTRIBUTES/VARIABLE_VALUEB+batch_norm3/beta/.ATTRIBUTES/VARIABLE_VALUEB2batch_norm3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB6batch_norm3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB'conv4/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv4/bias/.ATTRIBUTES/VARIABLE_VALUEB,batch_norm4/gamma/.ATTRIBUTES/VARIABLE_VALUEB+batch_norm4/beta/.ATTRIBUTES/VARIABLE_VALUEB2batch_norm4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB6batch_norm4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB'conv5/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv5/bias/.ATTRIBUTES/VARIABLE_VALUEB,batch_norm5/gamma/.ATTRIBUTES/VARIABLE_VALUEB+batch_norm5/beta/.ATTRIBUTES/VARIABLE_VALUEB2batch_norm5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB6batch_norm5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense2/bias/.ATTRIBUTES/VARIABLE_VALUEB,classifier/kernel/.ATTRIBUTES/VARIABLE_VALUEB*classifier/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHº
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ú
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ª
_output_shapes
:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp'assignvariableop_alex_net_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp'assignvariableop_1_alex_net_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_2AssignVariableOp5assignvariableop_2_alex_net_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_3AssignVariableOp4assignvariableop_3_alex_net_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_4AssignVariableOp;assignvariableop_4_alex_net_batch_normalization_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_5AssignVariableOp?assignvariableop_5_alex_net_batch_normalization_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp+assignvariableop_6_alex_net_conv2d_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp)assignvariableop_7_alex_net_conv2d_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_8AssignVariableOp7assignvariableop_8_alex_net_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_9AssignVariableOp6assignvariableop_9_alex_net_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_10AssignVariableOp>assignvariableop_10_alex_net_batch_normalization_1_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_11AssignVariableOpBassignvariableop_11_alex_net_batch_normalization_1_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp,assignvariableop_12_alex_net_conv2d_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp*assignvariableop_13_alex_net_conv2d_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_14AssignVariableOp8assignvariableop_14_alex_net_batch_normalization_2_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_15AssignVariableOp7assignvariableop_15_alex_net_batch_normalization_2_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_16AssignVariableOp>assignvariableop_16_alex_net_batch_normalization_2_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_17AssignVariableOpBassignvariableop_17_alex_net_batch_normalization_2_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp,assignvariableop_18_alex_net_conv2d_3_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp*assignvariableop_19_alex_net_conv2d_3_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_20AssignVariableOp8assignvariableop_20_alex_net_batch_normalization_3_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_21AssignVariableOp7assignvariableop_21_alex_net_batch_normalization_3_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_22AssignVariableOp>assignvariableop_22_alex_net_batch_normalization_3_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_23AssignVariableOpBassignvariableop_23_alex_net_batch_normalization_3_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp,assignvariableop_24_alex_net_conv2d_4_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp*assignvariableop_25_alex_net_conv2d_4_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_26AssignVariableOp8assignvariableop_26_alex_net_batch_normalization_4_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_27AssignVariableOp7assignvariableop_27_alex_net_batch_normalization_4_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_28AssignVariableOp>assignvariableop_28_alex_net_batch_normalization_4_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_29AssignVariableOpBassignvariableop_29_alex_net_batch_normalization_4_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp)assignvariableop_30_alex_net_dense_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp'assignvariableop_31_alex_net_dense_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp+assignvariableop_32_alex_net_dense_1_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp)assignvariableop_33_alex_net_dense_1_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp+assignvariableop_34_alex_net_dense_2_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp)assignvariableop_35_alex_net_dense_2_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ç
Identity_36Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_37IdentityIdentity_36:output:0^NoOp_1*
T0*
_output_shapes
: Ô
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_37Identity_37:output:0*]
_input_shapesL
J: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ó	
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_160236

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
	
Õ
6__inference_batch_normalization_3_layer_call_fn_161932

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_159729
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ô
C__inference_dense_2_layer_call_and_return_conditional_losses_160094

inputs0
matmul_readvariableop_resource: 
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ê
F
*__inference_rescaling_layer_call_fn_161577

inputs
identity½
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_rescaling_layer_call_and_return_conditional_losses_159862j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿãã:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã
 
_user_specified_nameinputs
×
¦!
D__inference_alex_net_layer_call_and_return_conditional_losses_161308

inputs?
%conv2d_conv2d_readvariableop_resource:`4
&conv2d_biasadd_readvariableop_resource:`9
+batch_normalization_readvariableop_resource:`;
-batch_normalization_readvariableop_1_resource:`J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:`L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:`B
'conv2d_1_conv2d_readvariableop_resource:`7
(conv2d_1_biasadd_readvariableop_resource:	<
-batch_normalization_1_readvariableop_resource:	>
/batch_normalization_1_readvariableop_1_resource:	M
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_2_conv2d_readvariableop_resource:7
(conv2d_2_biasadd_readvariableop_resource:	<
-batch_normalization_2_readvariableop_resource:	>
/batch_normalization_2_readvariableop_1_resource:	M
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_3_conv2d_readvariableop_resource:7
(conv2d_3_biasadd_readvariableop_resource:	<
-batch_normalization_3_readvariableop_resource:	>
/batch_normalization_3_readvariableop_1_resource:	M
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_4_conv2d_readvariableop_resource:7
(conv2d_4_biasadd_readvariableop_resource:	<
-batch_normalization_4_readvariableop_resource:	>
/batch_normalization_4_readvariableop_1_resource:	M
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	7
$dense_matmul_readvariableop_resource:	H 3
%dense_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource:  5
'dense_1_biasadd_readvariableop_resource: 8
&dense_2_matmul_readvariableop_resource: 
5
'dense_2_biasadd_readvariableop_resource:

identity¢8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp¢:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp¢:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp¢:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp¢:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp¢3batch_normalization/FusedBatchNormV3/ReadVariableOp¢5batch_normalization/FusedBatchNormV3/ReadVariableOp_1¢"batch_normalization/ReadVariableOp¢$batch_normalization/ReadVariableOp_1¢5batch_normalization_1/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_1/ReadVariableOp¢&batch_normalization_1/ReadVariableOp_1¢5batch_normalization_2/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_2/ReadVariableOp¢&batch_normalization_2/ReadVariableOp_1¢5batch_normalization_3/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_3/ReadVariableOp¢&batch_normalization_3/ReadVariableOp_1¢5batch_normalization_4/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_4/ReadVariableOp¢&batch_normalization_4/ReadVariableOp_1¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp¢conv2d_2/BiasAdd/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOp¢conv2d_3/BiasAdd/ReadVariableOp¢conv2d_3/Conv2D/ReadVariableOp¢conv2d_4/BiasAdd/ReadVariableOp¢conv2d_4/Conv2D/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOpU
rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;W
rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    s
rescaling/mulMulinputsrescaling/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã
rescaling/addAddV2rescaling/mul:z:0rescaling/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:`*
dtype0³
conv2d/Conv2DConv2Drescaling/add:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ77`*
paddingVALID*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ77`f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ77`
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:`*
dtype0
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:`*
dtype0¬
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0°
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0­
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ77`:`:`:`:`:*
epsilon%o:*
is_training( ·
max_pooling2d/MaxPoolMaxPool(batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
ksize
*
paddingVALID*
strides

conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:`*
dtype0Ä
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0¾
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ¼
max_pooling2d_1/MaxPoolMaxPool*batch_normalization_1/FusedBatchNormV3:y:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Æ
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0¾
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ð
conv2d_3/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0¾
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/Relu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ð
conv2d_4/Conv2DConv2D*batch_normalization_3/FusedBatchNormV3:y:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0¾
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/Relu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ¼
max_pooling2d_2/MaxPoolMaxPool*batch_normalization_4/FusedBatchNormV3:y:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ $  
flatten/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	H *
dtype0
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ h
dropout/IdentityIdentitydense/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ l
dropout_1/IdentityIdentitydense_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: 
*
dtype0
dense_2/MatMulMatMuldropout_1/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
f
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¦
8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:`*
dtype0¦
)alex_net/conv2d/kernel/Regularizer/SquareSquare@alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:`
(alex_net/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             °
&alex_net/conv2d/kernel/Regularizer/SumSum-alex_net/conv2d/kernel/Regularizer/Square:y:01alex_net/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: m
(alex_net/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;²
&alex_net/conv2d/kernel/Regularizer/mulMul1alex_net/conv2d/kernel/Regularizer/mul/x:output:0/alex_net/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: «
:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:`*
dtype0«
+alex_net/conv2d_1/kernel/Regularizer/SquareSquareBalex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:`
*alex_net/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_1/kernel/Regularizer/SumSum/alex_net/conv2d_1/kernel/Regularizer/Square:y:03alex_net/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_1/kernel/Regularizer/mulMul3alex_net/conv2d_1/kernel/Regularizer/mul/x:output:01alex_net/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¬
:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¬
+alex_net/conv2d_2/kernel/Regularizer/SquareSquareBalex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:
*alex_net/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_2/kernel/Regularizer/SumSum/alex_net/conv2d_2/kernel/Regularizer/Square:y:03alex_net/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_2/kernel/Regularizer/mulMul3alex_net/conv2d_2/kernel/Regularizer/mul/x:output:01alex_net/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¬
:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¬
+alex_net/conv2d_3/kernel/Regularizer/SquareSquareBalex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:
*alex_net/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_3/kernel/Regularizer/SumSum/alex_net/conv2d_3/kernel/Regularizer/Square:y:03alex_net/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_3/kernel/Regularizer/mulMul3alex_net/conv2d_3/kernel/Regularizer/mul/x:output:01alex_net/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¬
:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¬
+alex_net/conv2d_4/kernel/Regularizer/SquareSquareBalex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:
*alex_net/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_4/kernel/Regularizer/SumSum/alex_net/conv2d_4/kernel/Regularizer/Square:y:03alex_net/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_4/kernel/Regularizer/mulMul3alex_net/conv2d_4/kernel/Regularizer/mul/x:output:01alex_net/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Á
NoOpNoOp9^alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp;^alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp;^alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp;^alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp;^alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:ÿÿÿÿÿÿÿÿÿãã: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2t
8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp2x
:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2x
:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2x
:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2x
:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã
 
_user_specified_nameinputs
ô
¡
)__inference_conv2d_4_layer_call_fn_161996

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_160011x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Í
__inference_loss_fn_2_162243_
Calex_net_conv2d_2_kernel_regularizer_square_readvariableop_resource:
identity¢:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOpÈ
:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpCalex_net_conv2d_2_kernel_regularizer_square_readvariableop_resource*(
_output_shapes
:*
dtype0¬
+alex_net/conv2d_2/kernel/Regularizer/SquareSquareBalex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:
*alex_net/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_2/kernel/Regularizer/SumSum/alex_net/conv2d_2/kernel/Regularizer/Square:y:03alex_net/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_2/kernel/Regularizer/mulMul3alex_net/conv2d_2/kernel/Regularizer/mul/x:output:01alex_net/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentity,alex_net/conv2d_2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp;^alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2x
:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp
Ç
_
C__inference_flatten_layer_call_and_return_conditional_losses_162096

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ $  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿHY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿH"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Õ
6__inference_batch_normalization_2_layer_call_fn_161851

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_159696
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ
Ì
__inference_loss_fn_1_162232^
Calex_net_conv2d_1_kernel_regularizer_square_readvariableop_resource:`
identity¢:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOpÇ
:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpCalex_net_conv2d_1_kernel_regularizer_square_readvariableop_resource*'
_output_shapes
:`*
dtype0«
+alex_net/conv2d_1/kernel/Regularizer/SquareSquareBalex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:`
*alex_net/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_1/kernel/Regularizer/SumSum/alex_net/conv2d_1/kernel/Regularizer/Square:y:03alex_net/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_1/kernel/Regularizer/mulMul3alex_net/conv2d_1/kernel/Regularizer/mul/x:output:01alex_net/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentity,alex_net/conv2d_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp;^alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2x
:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp
Ø
¾
O__inference_batch_normalization_layer_call_and_return_conditional_losses_161679

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`:`:`:`:`:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
ê
Ä
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_161981

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê

O__inference_batch_normalization_layer_call_and_return_conditional_losses_159513

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`:`:`:`:`:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
Ü
 
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_159729

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ï
4__inference_batch_normalization_layer_call_fn_161643

inputs
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_159544
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
»
á
)__inference_alex_net_layer_call_fn_161060

inputs!
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
	unknown_3:`
	unknown_4:`$
	unknown_5:`
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	H 

unknown_30: 

unknown_31:  

unknown_32: 

unknown_33: 


unknown_34:

identity¢StatefulPartitionedCall¯
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_alex_net_layer_call_and_return_conditional_losses_160131o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:ÿÿÿÿÿÿÿÿÿãã: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã
 
_user_specified_nameinputs
ò
²$
D__inference_alex_net_layer_call_and_return_conditional_losses_161493

inputs?
%conv2d_conv2d_readvariableop_resource:`4
&conv2d_biasadd_readvariableop_resource:`9
+batch_normalization_readvariableop_resource:`;
-batch_normalization_readvariableop_1_resource:`J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:`L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:`B
'conv2d_1_conv2d_readvariableop_resource:`7
(conv2d_1_biasadd_readvariableop_resource:	<
-batch_normalization_1_readvariableop_resource:	>
/batch_normalization_1_readvariableop_1_resource:	M
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_2_conv2d_readvariableop_resource:7
(conv2d_2_biasadd_readvariableop_resource:	<
-batch_normalization_2_readvariableop_resource:	>
/batch_normalization_2_readvariableop_1_resource:	M
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_3_conv2d_readvariableop_resource:7
(conv2d_3_biasadd_readvariableop_resource:	<
-batch_normalization_3_readvariableop_resource:	>
/batch_normalization_3_readvariableop_1_resource:	M
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_4_conv2d_readvariableop_resource:7
(conv2d_4_biasadd_readvariableop_resource:	<
-batch_normalization_4_readvariableop_resource:	>
/batch_normalization_4_readvariableop_1_resource:	M
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	7
$dense_matmul_readvariableop_resource:	H 3
%dense_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource:  5
'dense_1_biasadd_readvariableop_resource: 8
&dense_2_matmul_readvariableop_resource: 
5
'dense_2_biasadd_readvariableop_resource:

identity¢8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp¢:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp¢:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp¢:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp¢:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp¢"batch_normalization/AssignNewValue¢$batch_normalization/AssignNewValue_1¢3batch_normalization/FusedBatchNormV3/ReadVariableOp¢5batch_normalization/FusedBatchNormV3/ReadVariableOp_1¢"batch_normalization/ReadVariableOp¢$batch_normalization/ReadVariableOp_1¢$batch_normalization_1/AssignNewValue¢&batch_normalization_1/AssignNewValue_1¢5batch_normalization_1/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_1/ReadVariableOp¢&batch_normalization_1/ReadVariableOp_1¢$batch_normalization_2/AssignNewValue¢&batch_normalization_2/AssignNewValue_1¢5batch_normalization_2/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_2/ReadVariableOp¢&batch_normalization_2/ReadVariableOp_1¢$batch_normalization_3/AssignNewValue¢&batch_normalization_3/AssignNewValue_1¢5batch_normalization_3/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_3/ReadVariableOp¢&batch_normalization_3/ReadVariableOp_1¢$batch_normalization_4/AssignNewValue¢&batch_normalization_4/AssignNewValue_1¢5batch_normalization_4/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_4/ReadVariableOp¢&batch_normalization_4/ReadVariableOp_1¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp¢conv2d_2/BiasAdd/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOp¢conv2d_3/BiasAdd/ReadVariableOp¢conv2d_3/Conv2D/ReadVariableOp¢conv2d_4/BiasAdd/ReadVariableOp¢conv2d_4/Conv2D/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOpU
rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;W
rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    s
rescaling/mulMulinputsrescaling/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã
rescaling/addAddV2rescaling/mul:z:0rescaling/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:`*
dtype0³
conv2d/Conv2DConv2Drescaling/add:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ77`*
paddingVALID*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ77`f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ77`
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:`*
dtype0
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:`*
dtype0¬
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0°
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0»
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ77`:`:`:`:`:*
epsilon%o:*
exponential_avg_factor%
×#<
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0·
max_pooling2d/MaxPoolMaxPool(batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
ksize
*
paddingVALID*
strides

conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:`*
dtype0Ä
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ì
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0¼
max_pooling2d_1/MaxPoolMaxPool*batch_normalization_1/FusedBatchNormV3:y:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Æ
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ì
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ð
conv2d_3/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ì
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/Relu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ð
conv2d_4/Conv2DConv2D*batch_normalization_3/FusedBatchNormV3:y:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ì
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/Relu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0¼
max_pooling2d_2/MaxPoolMaxPool*batch_normalization_4/FusedBatchNormV3:y:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ $  
flatten/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	H *
dtype0
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ]
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>¾
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?
dropout_1/dropout/MulMuldense_1/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
dropout_1/dropout/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
: 
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>Ä
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: 
*
dtype0
dense_2/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
f
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¦
8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:`*
dtype0¦
)alex_net/conv2d/kernel/Regularizer/SquareSquare@alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:`
(alex_net/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             °
&alex_net/conv2d/kernel/Regularizer/SumSum-alex_net/conv2d/kernel/Regularizer/Square:y:01alex_net/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: m
(alex_net/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;²
&alex_net/conv2d/kernel/Regularizer/mulMul1alex_net/conv2d/kernel/Regularizer/mul/x:output:0/alex_net/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: «
:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:`*
dtype0«
+alex_net/conv2d_1/kernel/Regularizer/SquareSquareBalex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:`
*alex_net/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_1/kernel/Regularizer/SumSum/alex_net/conv2d_1/kernel/Regularizer/Square:y:03alex_net/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_1/kernel/Regularizer/mulMul3alex_net/conv2d_1/kernel/Regularizer/mul/x:output:01alex_net/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¬
:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¬
+alex_net/conv2d_2/kernel/Regularizer/SquareSquareBalex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:
*alex_net/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_2/kernel/Regularizer/SumSum/alex_net/conv2d_2/kernel/Regularizer/Square:y:03alex_net/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_2/kernel/Regularizer/mulMul3alex_net/conv2d_2/kernel/Regularizer/mul/x:output:01alex_net/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¬
:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¬
+alex_net/conv2d_3/kernel/Regularizer/SquareSquareBalex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:
*alex_net/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_3/kernel/Regularizer/SumSum/alex_net/conv2d_3/kernel/Regularizer/Square:y:03alex_net/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_3/kernel/Regularizer/mulMul3alex_net/conv2d_3/kernel/Regularizer/mul/x:output:01alex_net/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¬
:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¬
+alex_net/conv2d_4/kernel/Regularizer/SquareSquareBalex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:
*alex_net/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_4/kernel/Regularizer/SumSum/alex_net/conv2d_4/kernel/Regularizer/Square:y:03alex_net/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_4/kernel/Regularizer/mulMul3alex_net/conv2d_4/kernel/Regularizer/mul/x:output:01alex_net/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Í
NoOpNoOp9^alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp;^alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp;^alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp;^alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp;^alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:ÿÿÿÿÿÿÿÿÿãã: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2t
8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp2x
:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2x
:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2x
:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2x
:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã
 
_user_specified_nameinputs


ô
C__inference_dense_2_layer_call_and_return_conditional_losses_162210

inputs0
matmul_readvariableop_resource: 
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ñ
 
)__inference_conv2d_1_layer_call_fn_161704

inputs"
unknown:`
	unknown_0:	
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_159914x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ`: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
Ê

O__inference_batch_normalization_layer_call_and_return_conditional_losses_161661

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`:`:`:`:`:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs


ô
C__inference_dense_1_layer_call_and_return_conditional_losses_162163

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ä
½
D__inference_conv2d_2_layer_call_and_return_conditional_losses_159947

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¬
+alex_net/conv2d_2/kernel/Regularizer/SquareSquareBalex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:
*alex_net/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_2/kernel/Regularizer/SumSum/alex_net/conv2d_2/kernel/Regularizer/Square:y:03alex_net/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_2/kernel/Regularizer/mulMul3alex_net/conv2d_2/kernel/Regularizer/mul/x:output:01alex_net/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp;^alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2x
:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_159844

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_159640

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_161689

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Õ
6__inference_batch_normalization_4_layer_call_fn_162039

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_159824
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

D
(__inference_dropout_layer_call_fn_162121

inputs
identity±
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_160057`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ä
½
D__inference_conv2d_4_layer_call_and_return_conditional_losses_162013

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¬
+alex_net/conv2d_4/kernel/Regularizer/SquareSquareBalex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:
*alex_net/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_4/kernel/Regularizer/SumSum/alex_net/conv2d_4/kernel/Regularizer/Square:y:03alex_net/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_4/kernel/Regularizer/mulMul3alex_net/conv2d_4/kernel/Regularizer/mul/x:output:01alex_net/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp;^alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2x
:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
 
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_159793

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
a
C__inference_dropout_layer_call_and_return_conditional_losses_162131

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

¶
B__inference_conv2d_layer_call_and_return_conditional_losses_159881

inputs8
conv2d_readvariableop_resource:`-
biasadd_readvariableop_resource:`
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:`*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ77`*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ77`X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ77`
8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:`*
dtype0¦
)alex_net/conv2d/kernel/Regularizer/SquareSquare@alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:`
(alex_net/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             °
&alex_net/conv2d/kernel/Regularizer/SumSum-alex_net/conv2d/kernel/Regularizer/Square:y:01alex_net/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: m
(alex_net/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;²
&alex_net/conv2d/kernel/Regularizer/mulMul1alex_net/conv2d/kernel/Regularizer/mul/x:output:0/alex_net/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ77`²
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp9^alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿãã: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2t
8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp8alex_net/conv2d/kernel/Regularizer/Square/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã
 
_user_specified_nameinputs
Ã

(__inference_dense_1_layer_call_fn_162152

inputs
unknown:  
	unknown_0: 
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_160070o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
	
Ï
4__inference_batch_normalization_layer_call_fn_161630

inputs
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_159513
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
Ü
 
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_159589

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä
½
D__inference_conv2d_4_layer_call_and_return_conditional_losses_160011

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¬
+alex_net/conv2d_4/kernel/Regularizer/SquareSquareBalex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:
*alex_net/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¶
(alex_net/conv2d_4/kernel/Regularizer/SumSum/alex_net/conv2d_4/kernel/Regularizer/Square:y:03alex_net/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: o
*alex_net/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;¸
(alex_net/conv2d_4/kernel/Regularizer/mulMul3alex_net/conv2d_4/kernel/Regularizer/mul/x:output:01alex_net/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp;^alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2x
:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:alex_net/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ó
A__inference_dense_layer_call_and_return_conditional_losses_162116

inputs1
matmul_readvariableop_resource:	H -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿH: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs
Ã

(__inference_dense_2_layer_call_fn_162199

inputs
unknown: 

	unknown_0:

identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_160094o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¾
â
)__inference_alex_net_layer_call_fn_160206
input_1!
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
	unknown_3:`
	unknown_4:`$
	unknown_5:`
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	H 

unknown_30: 

unknown_31:  

unknown_32: 

unknown_33: 


unknown_34:

identity¢StatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_alex_net_layer_call_and_return_conditional_losses_160131o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:ÿÿÿÿÿÿÿÿÿãã: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿãã
!
_user_specified_name	input_1
Â

&__inference_dense_layer_call_fn_162105

inputs
unknown:	H 
	unknown_0: 
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_160046o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿH: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*µ
serving_default¡
E
input_1:
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿãã<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:ºá
Ò
	scale
	conv1
batch_norm1
	pool1
	conv2
batch_norm2
	pool2
	conv3
	batch_norm3
	
conv4
batch_norm4
	conv5
batch_norm5
	pool3
flat

dense1
	drop1

dense2
	drop2

classifier
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_model
¥
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses"
_tf_keras_layer
»

#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
+axis
	,gamma
-beta
.moving_mean
/moving_variance
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
»

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
Daxis
	Egamma
Fbeta
Gmoving_mean
Hmoving_variance
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Ukernel
Vbias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
]axis
	^gamma
_beta
`moving_mean
amoving_variance
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses"
_tf_keras_layer
»

hkernel
ibias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
paxis
	qgamma
rbeta
smoving_mean
tmoving_variance
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses"
_tf_keras_layer
¾

{kernel
|bias
}	variables
~trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
+¡&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
¢	variables
£trainable_variables
¤regularization_losses
¥	keras_api
¦_random_generator
§__call__
+¨&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
©kernel
	ªbias
«	variables
¬trainable_variables
­regularization_losses
®	keras_api
¯__call__
+°&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
±	variables
²trainable_variables
³regularization_losses
´	keras_api
µ_random_generator
¶__call__
+·&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
¸kernel
	¹bias
º	variables
»trainable_variables
¼regularization_losses
½	keras_api
¾__call__
+¿&call_and_return_all_conditional_losses"
_tf_keras_layer
À
#0
$1
,2
-3
.4
/5
<6
=7
E8
F9
G10
H11
U12
V13
^14
_15
`16
a17
h18
i19
q20
r21
s22
t23
{24
|25
26
27
28
29
30
31
©32
ª33
¸34
¹35"
trackable_list_wrapper
î
#0
$1
,2
-3
<4
=5
E6
F7
U8
V9
^10
_11
h12
i13
q14
r15
{16
|17
18
19
20
21
©22
ª23
¸24
¹25"
trackable_list_wrapper
H
À0
Á1
Â2
Ã3
Ä4"
trackable_list_wrapper
Ï
Ånon_trainable_variables
Ælayers
Çmetrics
 Èlayer_regularization_losses
Élayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
æ2ã
)__inference_alex_net_layer_call_fn_160206
)__inference_alex_net_layer_call_fn_161060
)__inference_alex_net_layer_call_fn_161137
)__inference_alex_net_layer_call_fn_160701´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
D__inference_alex_net_layer_call_and_return_conditional_losses_161308
D__inference_alex_net_layer_call_and_return_conditional_losses_161493
D__inference_alex_net_layer_call_and_return_conditional_losses_160827
D__inference_alex_net_layer_call_and_return_conditional_losses_160953´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÌBÉ
!__inference__wrapped_model_159491input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
-
Êserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ënon_trainable_variables
Ìlayers
Ímetrics
 Îlayer_regularization_losses
Ïlayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_rescaling_layer_call_fn_161577¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_rescaling_layer_call_and_return_conditional_losses_161585¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
0:.`2alex_net/conv2d/kernel
": `2alex_net/conv2d/bias
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
(
À0"
trackable_list_wrapper
²
Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
Ñ2Î
'__inference_conv2d_layer_call_fn_161600¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_conv2d_layer_call_and_return_conditional_losses_161617¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
0:.`2"alex_net/batch_normalization/gamma
/:-`2!alex_net/batch_normalization/beta
8:6` (2(alex_net/batch_normalization/moving_mean
<::` (2,alex_net/batch_normalization/moving_variance
<
,0
-1
.2
/3"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Õnon_trainable_variables
Ölayers
×metrics
 Ølayer_regularization_losses
Ùlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
¦2£
4__inference_batch_normalization_layer_call_fn_161630
4__inference_batch_normalization_layer_call_fn_161643´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ü2Ù
O__inference_batch_normalization_layer_call_and_return_conditional_losses_161661
O__inference_batch_normalization_layer_call_and_return_conditional_losses_161679´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Únon_trainable_variables
Ûlayers
Ümetrics
 Ýlayer_regularization_losses
Þlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_max_pooling2d_layer_call_fn_161684¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_161689¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
3:1`2alex_net/conv2d_1/kernel
%:#2alex_net/conv2d_1/bias
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
(
Á0"
trackable_list_wrapper
²
ßnon_trainable_variables
àlayers
ámetrics
 âlayer_regularization_losses
ãlayer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_conv2d_1_layer_call_fn_161704¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_1_layer_call_and_return_conditional_losses_161721¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
3:12$alex_net/batch_normalization_1/gamma
2:02#alex_net/batch_normalization_1/beta
;:9 (2*alex_net/batch_normalization_1/moving_mean
?:= (2.alex_net/batch_normalization_1/moving_variance
<
E0
F1
G2
H3"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
änon_trainable_variables
ålayers
æmetrics
 çlayer_regularization_losses
èlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
ª2§
6__inference_batch_normalization_1_layer_call_fn_161734
6__inference_batch_normalization_1_layer_call_fn_161747´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
à2Ý
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_161765
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_161783´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
énon_trainable_variables
êlayers
ëmetrics
 ìlayer_regularization_losses
ílayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_max_pooling2d_1_layer_call_fn_161788¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_161793¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
4:22alex_net/conv2d_2/kernel
%:#2alex_net/conv2d_2/bias
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
(
Â0"
trackable_list_wrapper
²
înon_trainable_variables
ïlayers
ðmetrics
 ñlayer_regularization_losses
òlayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_conv2d_2_layer_call_fn_161808¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_2_layer_call_and_return_conditional_losses_161825¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
3:12$alex_net/batch_normalization_2/gamma
2:02#alex_net/batch_normalization_2/beta
;:9 (2*alex_net/batch_normalization_2/moving_mean
?:= (2.alex_net/batch_normalization_2/moving_variance
<
^0
_1
`2
a3"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ónon_trainable_variables
ôlayers
õmetrics
 ölayer_regularization_losses
÷layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
ª2§
6__inference_batch_normalization_2_layer_call_fn_161838
6__inference_batch_normalization_2_layer_call_fn_161851´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
à2Ý
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_161869
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_161887´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
4:22alex_net/conv2d_3/kernel
%:#2alex_net/conv2d_3/bias
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
(
Ã0"
trackable_list_wrapper
²
ønon_trainable_variables
ùlayers
úmetrics
 ûlayer_regularization_losses
ülayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_conv2d_3_layer_call_fn_161902¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_3_layer_call_and_return_conditional_losses_161919¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
3:12$alex_net/batch_normalization_3/gamma
2:02#alex_net/batch_normalization_3/beta
;:9 (2*alex_net/batch_normalization_3/moving_mean
?:= (2.alex_net/batch_normalization_3/moving_variance
<
q0
r1
s2
t3"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ýnon_trainable_variables
þlayers
ÿmetrics
 layer_regularization_losses
layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
ª2§
6__inference_batch_normalization_3_layer_call_fn_161932
6__inference_batch_normalization_3_layer_call_fn_161945´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
à2Ý
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_161963
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_161981´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
4:22alex_net/conv2d_4/kernel
%:#2alex_net/conv2d_4/bias
.
{0
|1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
(
Ä0"
trackable_list_wrapper
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
}	variables
~trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_conv2d_4_layer_call_fn_161996¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_4_layer_call_and_return_conditional_losses_162013¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
3:12$alex_net/batch_normalization_4/gamma
2:02#alex_net/batch_normalization_4/beta
;:9 (2*alex_net/batch_normalization_4/moving_mean
?:= (2.alex_net/batch_normalization_4/moving_variance
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ª2§
6__inference_batch_normalization_4_layer_call_fn_162026
6__inference_batch_normalization_4_layer_call_fn_162039´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
à2Ý
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_162057
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_162075´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_max_pooling2d_2_layer_call_fn_162080¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_162085¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_flatten_layer_call_fn_162090¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_flatten_layer_call_and_return_conditional_losses_162096¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
(:&	H 2alex_net/dense/kernel
!: 2alex_net/dense/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
+¡&call_and_return_all_conditional_losses
'¡"call_and_return_conditional_losses"
_generic_user_object
Ð2Í
&__inference_dense_layer_call_fn_162105¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_dense_layer_call_and_return_conditional_losses_162116¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¢	variables
£trainable_variables
¤regularization_losses
§__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
(__inference_dropout_layer_call_fn_162121
(__inference_dropout_layer_call_fn_162126´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ä2Á
C__inference_dropout_layer_call_and_return_conditional_losses_162131
C__inference_dropout_layer_call_and_return_conditional_losses_162143´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
):'  2alex_net/dense_1/kernel
#:! 2alex_net/dense_1/bias
0
©0
ª1"
trackable_list_wrapper
0
©0
ª1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
 non_trainable_variables
¡layers
¢metrics
 £layer_regularization_losses
¤layer_metrics
«	variables
¬trainable_variables
­regularization_losses
¯__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_dense_1_layer_call_fn_162152¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_1_layer_call_and_return_conditional_losses_162163¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¥non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
±	variables
²trainable_variables
³regularization_losses
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
*__inference_dropout_1_layer_call_fn_162168
*__inference_dropout_1_layer_call_fn_162173´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_dropout_1_layer_call_and_return_conditional_losses_162178
E__inference_dropout_1_layer_call_and_return_conditional_losses_162190´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
):' 
2alex_net/dense_2/kernel
#:!
2alex_net/dense_2/bias
0
¸0
¹1"
trackable_list_wrapper
0
¸0
¹1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ªnon_trainable_variables
«layers
¬metrics
 ­layer_regularization_losses
®layer_metrics
º	variables
»trainable_variables
¼regularization_losses
¾__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_dense_2_layer_call_fn_162199¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_2_layer_call_and_return_conditional_losses_162210¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
³2°
__inference_loss_fn_0_162221
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_1_162232
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_2_162243
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_3_162254
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_4_162265
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
h
.0
/1
G2
H3
`4
a5
s6
t7
8
9"
trackable_list_wrapper
¶
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ËBÈ
$__inference_signature_wrapper_161572input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
À0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
Á0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
Â0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
Ã0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
Ä0"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapperÇ
!__inference__wrapped_model_159491¡.#$,-./<=EFGHUV^_`ahiqrst{|©ª¸¹:¢7
0¢-
+(
input_1ÿÿÿÿÿÿÿÿÿãã
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
à
D__inference_alex_net_layer_call_and_return_conditional_losses_160827.#$,-./<=EFGHUV^_`ahiqrst{|©ª¸¹>¢;
4¢1
+(
input_1ÿÿÿÿÿÿÿÿÿãã
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 à
D__inference_alex_net_layer_call_and_return_conditional_losses_160953.#$,-./<=EFGHUV^_`ahiqrst{|©ª¸¹>¢;
4¢1
+(
input_1ÿÿÿÿÿÿÿÿÿãã
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ß
D__inference_alex_net_layer_call_and_return_conditional_losses_161308.#$,-./<=EFGHUV^_`ahiqrst{|©ª¸¹=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿãã
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ß
D__inference_alex_net_layer_call_and_return_conditional_losses_161493.#$,-./<=EFGHUV^_`ahiqrst{|©ª¸¹=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿãã
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ¸
)__inference_alex_net_layer_call_fn_160206.#$,-./<=EFGHUV^_`ahiqrst{|©ª¸¹>¢;
4¢1
+(
input_1ÿÿÿÿÿÿÿÿÿãã
p 
ª "ÿÿÿÿÿÿÿÿÿ
¸
)__inference_alex_net_layer_call_fn_160701.#$,-./<=EFGHUV^_`ahiqrst{|©ª¸¹>¢;
4¢1
+(
input_1ÿÿÿÿÿÿÿÿÿãã
p
ª "ÿÿÿÿÿÿÿÿÿ
·
)__inference_alex_net_layer_call_fn_161060.#$,-./<=EFGHUV^_`ahiqrst{|©ª¸¹=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿãã
p 
ª "ÿÿÿÿÿÿÿÿÿ
·
)__inference_alex_net_layer_call_fn_161137.#$,-./<=EFGHUV^_`ahiqrst{|©ª¸¹=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿãã
p
ª "ÿÿÿÿÿÿÿÿÿ
î
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_161765EFGHN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 î
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_161783EFGHN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
6__inference_batch_normalization_1_layer_call_fn_161734EFGHN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÆ
6__inference_batch_normalization_1_layer_call_fn_161747EFGHN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_161869^_`aN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 î
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_161887^_`aN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
6__inference_batch_normalization_2_layer_call_fn_161838^_`aN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÆ
6__inference_batch_normalization_2_layer_call_fn_161851^_`aN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_161963qrstN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 î
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_161981qrstN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
6__inference_batch_normalization_3_layer_call_fn_161932qrstN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÆ
6__inference_batch_normalization_3_layer_call_fn_161945qrstN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿò
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_162057N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ò
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_162075N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ê
6__inference_batch_normalization_4_layer_call_fn_162026N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÊ
6__inference_batch_normalization_4_layer_call_fn_162039N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿê
O__inference_batch_normalization_layer_call_and_return_conditional_losses_161661,-./M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
 ê
O__inference_batch_normalization_layer_call_and_return_conditional_losses_161679,-./M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
 Â
4__inference_batch_normalization_layer_call_fn_161630,-./M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`Â
4__inference_batch_normalization_layer_call_fn_161643,-./M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`µ
D__inference_conv2d_1_layer_call_and_return_conditional_losses_161721m<=7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ`
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
)__inference_conv2d_1_layer_call_fn_161704`<=7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ`
ª "!ÿÿÿÿÿÿÿÿÿ¶
D__inference_conv2d_2_layer_call_and_return_conditional_losses_161825nUV8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
)__inference_conv2d_2_layer_call_fn_161808aUV8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ¶
D__inference_conv2d_3_layer_call_and_return_conditional_losses_161919nhi8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
)__inference_conv2d_3_layer_call_fn_161902ahi8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ¶
D__inference_conv2d_4_layer_call_and_return_conditional_losses_162013n{|8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
)__inference_conv2d_4_layer_call_fn_161996a{|8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ´
B__inference_conv2d_layer_call_and_return_conditional_losses_161617n#$9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿãã
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ77`
 
'__inference_conv2d_layer_call_fn_161600a#$9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿãã
ª " ÿÿÿÿÿÿÿÿÿ77`¥
C__inference_dense_1_layer_call_and_return_conditional_losses_162163^©ª/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 }
(__inference_dense_1_layer_call_fn_162152Q©ª/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ¥
C__inference_dense_2_layer_call_and_return_conditional_losses_162210^¸¹/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 }
(__inference_dense_2_layer_call_fn_162199Q¸¹/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ
¤
A__inference_dense_layer_call_and_return_conditional_losses_162116_0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿH
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 |
&__inference_dense_layer_call_fn_162105R0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿH
ª "ÿÿÿÿÿÿÿÿÿ ¥
E__inference_dropout_1_layer_call_and_return_conditional_losses_162178\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 ¥
E__inference_dropout_1_layer_call_and_return_conditional_losses_162190\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 }
*__inference_dropout_1_layer_call_fn_162168O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "ÿÿÿÿÿÿÿÿÿ }
*__inference_dropout_1_layer_call_fn_162173O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "ÿÿÿÿÿÿÿÿÿ £
C__inference_dropout_layer_call_and_return_conditional_losses_162131\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 £
C__inference_dropout_layer_call_and_return_conditional_losses_162143\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 {
(__inference_dropout_layer_call_fn_162121O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "ÿÿÿÿÿÿÿÿÿ {
(__inference_dropout_layer_call_fn_162126O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "ÿÿÿÿÿÿÿÿÿ ©
C__inference_flatten_layer_call_and_return_conditional_losses_162096b8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿH
 
(__inference_flatten_layer_call_fn_162090U8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿH;
__inference_loss_fn_0_162221#¢

¢ 
ª " ;
__inference_loss_fn_1_162232<¢

¢ 
ª " ;
__inference_loss_fn_2_162243U¢

¢ 
ª " ;
__inference_loss_fn_3_162254h¢

¢ 
ª " ;
__inference_loss_fn_4_162265{¢

¢ 
ª " î
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_161793R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_1_layer_call_fn_161788R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_162085R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_2_layer_call_fn_162080R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿì
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_161689R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_max_pooling2d_layer_call_fn_161684R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿµ
E__inference_rescaling_layer_call_and_return_conditional_losses_161585l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿãã
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿãã
 
*__inference_rescaling_layer_call_fn_161577_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿãã
ª ""ÿÿÿÿÿÿÿÿÿããÕ
$__inference_signature_wrapper_161572¬.#$,-./<=EFGHUV^_`ahiqrst{|©ª¸¹E¢B
¢ 
;ª8
6
input_1+(
input_1ÿÿÿÿÿÿÿÿÿãã"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
