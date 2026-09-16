-- Ported from the throwaway table-decoder spike (empty-cone); 24-token chunks and their stack-0 compositions.
import Slang.Generated.SourceWitness.Decoder.ChunksA

set_option autoImplicit false
set_option warningAsError true
set_option linter.missingDocs false

namespace Slang.Generated.SourceWitness.Decoder

open Slang Slang.Decode Slang.Source Slang.Source.Parsing

def tableL744 : List Token :=
  [{ kind := .comma, span := ⟨3899, 3900⟩ },
  { kind := .ident "mask", span := ⟨3901, 3905⟩ },
  { kind := .comma, span := ⟨3905, 3906⟩ },
  { kind := .ident "key", span := ⟨3907, 3910⟩ },
  { kind := .comma, span := ⟨3910, 3911⟩ },
  { kind := .ident "slot", span := ⟨3912, 3916⟩ },
  { kind := .comma, span := ⟨3916, 3917⟩ },
  { kind := .ident "step", span := ⟨3918, 3922⟩ },
  { kind := .rparen, span := ⟨3922, 3923⟩ },
  { kind := .lbrace, span := ⟨3924, 3925⟩ },
  { kind := .newline, span := ⟨3925, 3926⟩ },
  { kind := .ident "if", span := ⟨3928, 3930⟩ },
  { kind := .ident "step", span := ⟨3931, 3935⟩ },
  { kind := .ge, span := ⟨3936, 3938⟩ },
  { kind := .ident "cap", span := ⟨3939, 3942⟩ },
  { kind := .lbrace, span := ⟨3943, 3944⟩ },
  { kind := .ident "return", span := ⟨3945, 3951⟩ },
  { kind := .numeral 0, span := ⟨3952, 3953⟩ },
  { kind := .other (Char.ofNat 45), span := ⟨3954, 3955⟩ },
  { kind := .numeral 1, span := ⟨3956, 3957⟩ },
  { kind := .semi, span := ⟨3957, 3958⟩ },
  { kind := .rbrace, span := ⟨3959, 3960⟩ },
  { kind := .newline, span := ⟨3960, 3961⟩ },
  { kind := .ident "currSlot", span := ⟨3963, 3971⟩ }]

def tableL768 : List Token :=
  [{ kind := .assign, span := ⟨3972, 3973⟩ },
  { kind := .lparen, span := ⟨3974, 3975⟩ },
  { kind := .ident "slot", span := ⟨3975, 3979⟩ },
  { kind := .other (Char.ofNat 43), span := ⟨3980, 3981⟩ },
  { kind := .ident "step", span := ⟨3982, 3986⟩ },
  { kind := .rparen, span := ⟨3986, 3987⟩ },
  { kind := .ident "and", span := ⟨3988, 3991⟩ },
  { kind := .ident "mask", span := ⟨3992, 3996⟩ },
  { kind := .semi, span := ⟨3996, 3997⟩ },
  { kind := .newline, span := ⟨3997, 3998⟩ },
  { kind := .ident "slotKey", span := ⟨4000, 4007⟩ },
  { kind := .assign, span := ⟨4008, 4009⟩ },
  { kind := .ident "tableSlotKey", span := ⟨4010, 4022⟩ },
  { kind := .lparen, span := ⟨4022, 4023⟩ },
  { kind := .ident "base", span := ⟨4023, 4027⟩ },
  { kind := .comma, span := ⟨4027, 4028⟩ },
  { kind := .ident "currSlot", span := ⟨4029, 4037⟩ },
  { kind := .rparen, span := ⟨4037, 4038⟩ },
  { kind := .semi, span := ⟨4038, 4039⟩ },
  { kind := .newline, span := ⟨4039, 4040⟩ },
  { kind := .ident "if", span := ⟨4042, 4044⟩ },
  { kind := .ident "slotKey", span := ⟨4045, 4052⟩ },
  { kind := .ident "is", span := ⟨4053, 4055⟩ },
  { kind := .ident "TABLE_EMPTY", span := ⟨4056, 4067⟩ }]

def tableL792 : List Token :=
  [{ kind := .lbrace, span := ⟨4068, 4069⟩ },
  { kind := .ident "return", span := ⟨4070, 4076⟩ },
  { kind := .numeral 0, span := ⟨4077, 4078⟩ },
  { kind := .other (Char.ofNat 45), span := ⟨4079, 4080⟩ },
  { kind := .numeral 1, span := ⟨4081, 4082⟩ },
  { kind := .semi, span := ⟨4082, 4083⟩ },
  { kind := .rbrace, span := ⟨4084, 4085⟩ },
  { kind := .newline, span := ⟨4085, 4086⟩ },
  { kind := .ident "if", span := ⟨4088, 4090⟩ },
  { kind := .ident "slotKey", span := ⟨4091, 4098⟩ },
  { kind := .ident "is", span := ⟨4099, 4101⟩ },
  { kind := .ident "key", span := ⟨4102, 4105⟩ },
  { kind := .lbrace, span := ⟨4106, 4107⟩ },
  { kind := .ident "return", span := ⟨4108, 4114⟩ },
  { kind := .ident "currSlot", span := ⟨4115, 4123⟩ },
  { kind := .semi, span := ⟨4123, 4124⟩ },
  { kind := .rbrace, span := ⟨4125, 4126⟩ },
  { kind := .newline, span := ⟨4126, 4127⟩ },
  { kind := .ident "return", span := ⟨4129, 4135⟩ },
  { kind := .ident "tableLookupSlotGo", span := ⟨4136, 4153⟩ },
  { kind := .lparen, span := ⟨4153, 4154⟩ },
  { kind := .ident "base", span := ⟨4154, 4158⟩ },
  { kind := .comma, span := ⟨4158, 4159⟩ },
  { kind := .ident "cap", span := ⟨4160, 4163⟩ }]

def tableL816 : List Token :=
  [{ kind := .comma, span := ⟨4163, 4164⟩ },
  { kind := .ident "mask", span := ⟨4165, 4169⟩ },
  { kind := .comma, span := ⟨4169, 4170⟩ },
  { kind := .ident "key", span := ⟨4171, 4174⟩ },
  { kind := .comma, span := ⟨4174, 4175⟩ },
  { kind := .ident "slot", span := ⟨4176, 4180⟩ },
  { kind := .comma, span := ⟨4180, 4181⟩ },
  { kind := .ident "step", span := ⟨4182, 4186⟩ },
  { kind := .other (Char.ofNat 43), span := ⟨4187, 4188⟩ },
  { kind := .numeral 1, span := ⟨4189, 4190⟩ },
  { kind := .rparen, span := ⟨4190, 4191⟩ },
  { kind := .semi, span := ⟨4191, 4192⟩ },
  { kind := .newline, span := ⟨4192, 4193⟩ },
  { kind := .rbrace, span := ⟨4193, 4194⟩ },
  { kind := .newline, span := ⟨4194, 4195⟩ },
  { kind := .newline, span := ⟨4195, 4196⟩ },
  { kind := .newline, span := ⟨4251, 4252⟩ },
  { kind := .ident "tableFind", span := ⟨4252, 4261⟩ },
  { kind := .lparen, span := ⟨4261, 4262⟩ },
  { kind := .ident "base", span := ⟨4262, 4266⟩ },
  { kind := .comma, span := ⟨4266, 4267⟩ },
  { kind := .ident "cap", span := ⟨4268, 4271⟩ },
  { kind := .comma, span := ⟨4271, 4272⟩ },
  { kind := .ident "key", span := ⟨4273, 4276⟩ }]

def tableL840 : List Token :=
  [{ kind := .rparen, span := ⟨4276, 4277⟩ },
  { kind := .lbrace, span := ⟨4278, 4279⟩ },
  { kind := .newline, span := ⟨4279, 4280⟩ },
  { kind := .ident "if", span := ⟨4282, 4284⟩ },
  { kind := .ident "tableCapacityOk", span := ⟨4285, 4300⟩ },
  { kind := .lparen, span := ⟨4300, 4301⟩ },
  { kind := .ident "cap", span := ⟨4301, 4304⟩ },
  { kind := .rparen, span := ⟨4304, 4305⟩ },
  { kind := .ident "is", span := ⟨4306, 4308⟩ },
  { kind := .numeral 0, span := ⟨4309, 4310⟩ },
  { kind := .lbrace, span := ⟨4311, 4312⟩ },
  { kind := .ident "return", span := ⟨4313, 4319⟩ },
  { kind := .ident "TABLE_EMPTY", span := ⟨4320, 4331⟩ },
  { kind := .semi, span := ⟨4331, 4332⟩ },
  { kind := .rbrace, span := ⟨4333, 4334⟩ },
  { kind := .newline, span := ⟨4334, 4335⟩ },
  { kind := .ident "if", span := ⟨4337, 4339⟩ },
  { kind := .ident "key", span := ⟨4340, 4343⟩ },
  { kind := .ident "is", span := ⟨4344, 4346⟩ },
  { kind := .ident "TABLE_EMPTY", span := ⟨4347, 4358⟩ },
  { kind := .lbrace, span := ⟨4359, 4360⟩ },
  { kind := .ident "return", span := ⟨4361, 4367⟩ },
  { kind := .ident "TABLE_EMPTY", span := ⟨4368, 4379⟩ },
  { kind := .semi, span := ⟨4379, 4380⟩ }]

def tableL864 : List Token :=
  [{ kind := .rbrace, span := ⟨4381, 4382⟩ },
  { kind := .newline, span := ⟨4382, 4383⟩ },
  { kind := .ident "mask", span := ⟨4385, 4389⟩ },
  { kind := .assign, span := ⟨4390, 4391⟩ },
  { kind := .ident "cap", span := ⟨4392, 4395⟩ },
  { kind := .other (Char.ofNat 45), span := ⟨4396, 4397⟩ },
  { kind := .numeral 1, span := ⟨4398, 4399⟩ },
  { kind := .semi, span := ⟨4399, 4400⟩ },
  { kind := .newline, span := ⟨4400, 4401⟩ },
  { kind := .ident "initSlot", span := ⟨4403, 4411⟩ },
  { kind := .assign, span := ⟨4412, 4413⟩ },
  { kind := .ident "tableSpread", span := ⟨4414, 4425⟩ },
  { kind := .lparen, span := ⟨4425, 4426⟩ },
  { kind := .ident "key", span := ⟨4426, 4429⟩ },
  { kind := .rparen, span := ⟨4429, 4430⟩ },
  { kind := .ident "and", span := ⟨4431, 4434⟩ },
  { kind := .ident "mask", span := ⟨4435, 4439⟩ },
  { kind := .semi, span := ⟨4439, 4440⟩ },
  { kind := .newline, span := ⟨4440, 4441⟩ },
  { kind := .ident "return", span := ⟨4443, 4449⟩ },
  { kind := .ident "tableLookupSlotGo", span := ⟨4450, 4467⟩ },
  { kind := .lparen, span := ⟨4467, 4468⟩ },
  { kind := .ident "base", span := ⟨4468, 4472⟩ },
  { kind := .comma, span := ⟨4472, 4473⟩ }]

def tableL888 : List Token :=
  [{ kind := .ident "cap", span := ⟨4474, 4477⟩ },
  { kind := .comma, span := ⟨4477, 4478⟩ },
  { kind := .ident "mask", span := ⟨4479, 4483⟩ },
  { kind := .comma, span := ⟨4483, 4484⟩ },
  { kind := .ident "key", span := ⟨4485, 4488⟩ },
  { kind := .comma, span := ⟨4488, 4489⟩ },
  { kind := .ident "initSlot", span := ⟨4490, 4498⟩ },
  { kind := .comma, span := ⟨4498, 4499⟩ },
  { kind := .numeral 0, span := ⟨4500, 4501⟩ },
  { kind := .rparen, span := ⟨4501, 4502⟩ },
  { kind := .semi, span := ⟨4502, 4503⟩ },
  { kind := .newline, span := ⟨4503, 4504⟩ },
  { kind := .rbrace, span := ⟨4504, 4505⟩ },
  { kind := .newline, span := ⟨4505, 4506⟩ },
  { kind := .newline, span := ⟨4506, 4507⟩ },
  { kind := .newline, span := ⟨4562, 4563⟩ },
  { kind := .ident "tableGet", span := ⟨4563, 4571⟩ },
  { kind := .lparen, span := ⟨4571, 4572⟩ },
  { kind := .ident "base", span := ⟨4572, 4576⟩ },
  { kind := .comma, span := ⟨4576, 4577⟩ },
  { kind := .ident "cap", span := ⟨4578, 4581⟩ },
  { kind := .comma, span := ⟨4581, 4582⟩ },
  { kind := .ident "key", span := ⟨4583, 4586⟩ },
  { kind := .rparen, span := ⟨4586, 4587⟩ }]

def tableL912 : List Token :=
  [{ kind := .lbrace, span := ⟨4588, 4589⟩ },
  { kind := .newline, span := ⟨4589, 4590⟩ },
  { kind := .ident "slot", span := ⟨4592, 4596⟩ },
  { kind := .assign, span := ⟨4597, 4598⟩ },
  { kind := .ident "tableFind", span := ⟨4599, 4608⟩ },
  { kind := .lparen, span := ⟨4608, 4609⟩ },
  { kind := .ident "base", span := ⟨4609, 4613⟩ },
  { kind := .comma, span := ⟨4613, 4614⟩ },
  { kind := .ident "cap", span := ⟨4615, 4618⟩ },
  { kind := .comma, span := ⟨4618, 4619⟩ },
  { kind := .ident "key", span := ⟨4620, 4623⟩ },
  { kind := .rparen, span := ⟨4623, 4624⟩ },
  { kind := .semi, span := ⟨4624, 4625⟩ },
  { kind := .newline, span := ⟨4625, 4626⟩ },
  { kind := .ident "if", span := ⟨4628, 4630⟩ },
  { kind := .ident "slot", span := ⟨4631, 4635⟩ },
  { kind := .lt, span := ⟨4636, 4637⟩ },
  { kind := .numeral 0, span := ⟨4638, 4639⟩ },
  { kind := .lbrace, span := ⟨4640, 4641⟩ },
  { kind := .ident "return", span := ⟨4642, 4648⟩ },
  { kind := .ident "TABLE_EMPTY", span := ⟨4649, 4660⟩ },
  { kind := .semi, span := ⟨4660, 4661⟩ },
  { kind := .rbrace, span := ⟨4662, 4663⟩ },
  { kind := .newline, span := ⟨4663, 4664⟩ }]

def tableL936 : List Token :=
  [{ kind := .ident "return", span := ⟨4666, 4672⟩ },
  { kind := .ident "tableSlotVal", span := ⟨4673, 4685⟩ },
  { kind := .lparen, span := ⟨4685, 4686⟩ },
  { kind := .ident "base", span := ⟨4686, 4690⟩ },
  { kind := .comma, span := ⟨4690, 4691⟩ },
  { kind := .ident "slot", span := ⟨4692, 4696⟩ },
  { kind := .rparen, span := ⟨4696, 4697⟩ },
  { kind := .semi, span := ⟨4697, 4698⟩ },
  { kind := .newline, span := ⟨4698, 4699⟩ },
  { kind := .rbrace, span := ⟨4699, 4700⟩ },
  { kind := .newline, span := ⟨4700, 4701⟩ },
  { kind := .newline, span := ⟨4701, 4702⟩ },
  { kind := .newline, span := ⟨4762, 4763⟩ },
  { kind := .ident "tableInsertWith", span := ⟨4763, 4778⟩ },
  { kind := .lparen, span := ⟨4778, 4779⟩ },
  { kind := .ident "base", span := ⟨4779, 4783⟩ },
  { kind := .comma, span := ⟨4783, 4784⟩ },
  { kind := .ident "cap", span := ⟨4785, 4788⟩ },
  { kind := .comma, span := ⟨4788, 4789⟩ },
  { kind := .ident "key", span := ⟨4790, 4793⟩ },
  { kind := .comma, span := ⟨4793, 4794⟩ },
  { kind := .ident "value", span := ⟨4795, 4800⟩ },
  { kind := .comma, span := ⟨4800, 4801⟩ },
  { kind := .ident "merge", span := ⟨4802, 4807⟩ }]

def tableL960 : List Token :=
  [{ kind := .rparen, span := ⟨4807, 4808⟩ },
  { kind := .lbrace, span := ⟨4809, 4810⟩ },
  { kind := .newline, span := ⟨4810, 4811⟩ },
  { kind := .ident "if", span := ⟨4813, 4815⟩ },
  { kind := .ident "tableCapacityOk", span := ⟨4816, 4831⟩ },
  { kind := .lparen, span := ⟨4831, 4832⟩ },
  { kind := .ident "cap", span := ⟨4832, 4835⟩ },
  { kind := .rparen, span := ⟨4835, 4836⟩ },
  { kind := .ident "is", span := ⟨4837, 4839⟩ },
  { kind := .numeral 0, span := ⟨4840, 4841⟩ },
  { kind := .lbrace, span := ⟨4842, 4843⟩ },
  { kind := .ident "return", span := ⟨4844, 4850⟩ },
  { kind := .ident "TABLE_EMPTY", span := ⟨4851, 4862⟩ },
  { kind := .semi, span := ⟨4862, 4863⟩ },
  { kind := .rbrace, span := ⟨4864, 4865⟩ },
  { kind := .newline, span := ⟨4865, 4866⟩ },
  { kind := .ident "if", span := ⟨4868, 4870⟩ },
  { kind := .ident "key", span := ⟨4871, 4874⟩ },
  { kind := .ident "is", span := ⟨4875, 4877⟩ },
  { kind := .ident "TABLE_EMPTY", span := ⟨4878, 4889⟩ },
  { kind := .lbrace, span := ⟨4890, 4891⟩ },
  { kind := .ident "return", span := ⟨4892, 4898⟩ },
  { kind := .ident "TABLE_EMPTY", span := ⟨4899, 4910⟩ },
  { kind := .semi, span := ⟨4910, 4911⟩ }]

def tableL984 : List Token :=
  [{ kind := .rbrace, span := ⟨4912, 4913⟩ },
  { kind := .newline, span := ⟨4913, 4914⟩ },
  { kind := .ident "if", span := ⟨4916, 4918⟩ },
  { kind := .lparen, span := ⟨4919, 4920⟩ },
  { kind := .ident "merge", span := ⟨4920, 4925⟩ },
  { kind := .lt, span := ⟨4926, 4927⟩ },
  { kind := .ident "MERGE_SET", span := ⟨4928, 4937⟩ },
  { kind := .rparen, span := ⟨4937, 4938⟩ },
  { kind := .ident "or", span := ⟨4939, 4941⟩ },
  { kind := .lparen, span := ⟨4942, 4943⟩ },
  { kind := .ident "merge", span := ⟨4943, 4948⟩ },
  { kind := .gt, span := ⟨4949, 4950⟩ },
  { kind := .ident "MERGE_ACC", span := ⟨4951, 4960⟩ },
  { kind := .rparen, span := ⟨4960, 4961⟩ },
  { kind := .lbrace, span := ⟨4962, 4963⟩ },
  { kind := .ident "return", span := ⟨4964, 4970⟩ },
  { kind := .ident "TABLE_EMPTY", span := ⟨4971, 4982⟩ },
  { kind := .semi, span := ⟨4982, 4983⟩ },
  { kind := .rbrace, span := ⟨4984, 4985⟩ },
  { kind := .newline, span := ⟨4985, 4986⟩ },
  { kind := .ident "mask", span := ⟨4988, 4992⟩ },
  { kind := .assign, span := ⟨4993, 4994⟩ },
  { kind := .ident "cap", span := ⟨4995, 4998⟩ },
  { kind := .other (Char.ofNat 45), span := ⟨4999, 5000⟩ }]

def tableL1008 : List Token :=
  [{ kind := .numeral 1, span := ⟨5001, 5002⟩ },
  { kind := .semi, span := ⟨5002, 5003⟩ },
  { kind := .newline, span := ⟨5003, 5004⟩ },
  { kind := .ident "initSlot", span := ⟨5006, 5014⟩ },
  { kind := .assign, span := ⟨5015, 5016⟩ },
  { kind := .ident "tableSpread", span := ⟨5017, 5028⟩ },
  { kind := .lparen, span := ⟨5028, 5029⟩ },
  { kind := .ident "key", span := ⟨5029, 5032⟩ },
  { kind := .rparen, span := ⟨5032, 5033⟩ },
  { kind := .ident "and", span := ⟨5034, 5037⟩ },
  { kind := .ident "mask", span := ⟨5038, 5042⟩ },
  { kind := .semi, span := ⟨5042, 5043⟩ },
  { kind := .newline, span := ⟨5043, 5044⟩ },
  { kind := .ident "slot", span := ⟨5046, 5050⟩ },
  { kind := .assign, span := ⟨5051, 5052⟩ },
  { kind := .ident "tableFindSlotGo", span := ⟨5053, 5068⟩ },
  { kind := .lparen, span := ⟨5068, 5069⟩ },
  { kind := .ident "base", span := ⟨5069, 5073⟩ },
  { kind := .comma, span := ⟨5073, 5074⟩ },
  { kind := .ident "cap", span := ⟨5075, 5078⟩ },
  { kind := .comma, span := ⟨5078, 5079⟩ },
  { kind := .ident "mask", span := ⟨5080, 5084⟩ },
  { kind := .comma, span := ⟨5084, 5085⟩ },
  { kind := .ident "key", span := ⟨5086, 5089⟩ }]

def tableL1032 : List Token :=
  [{ kind := .comma, span := ⟨5089, 5090⟩ },
  { kind := .ident "initSlot", span := ⟨5091, 5099⟩ },
  { kind := .comma, span := ⟨5099, 5100⟩ },
  { kind := .numeral 0, span := ⟨5101, 5102⟩ },
  { kind := .rparen, span := ⟨5102, 5103⟩ },
  { kind := .semi, span := ⟨5103, 5104⟩ },
  { kind := .newline, span := ⟨5104, 5105⟩ },
  { kind := .ident "if", span := ⟨5107, 5109⟩ },
  { kind := .ident "slot", span := ⟨5110, 5114⟩ },
  { kind := .lt, span := ⟨5115, 5116⟩ },
  { kind := .numeral 0, span := ⟨5117, 5118⟩ },
  { kind := .lbrace, span := ⟨5119, 5120⟩ },
  { kind := .ident "return", span := ⟨5121, 5127⟩ },
  { kind := .numeral 0, span := ⟨5128, 5129⟩ },
  { kind := .other (Char.ofNat 45), span := ⟨5130, 5131⟩ },
  { kind := .numeral 1, span := ⟨5132, 5133⟩ },
  { kind := .semi, span := ⟨5133, 5134⟩ },
  { kind := .rbrace, span := ⟨5135, 5136⟩ },
  { kind := .newline, span := ⟨5136, 5137⟩ },
  { kind := .ident "oldKey", span := ⟨5139, 5145⟩ },
  { kind := .assign, span := ⟨5146, 5147⟩ },
  { kind := .ident "tableSlotKey", span := ⟨5148, 5160⟩ },
  { kind := .lparen, span := ⟨5160, 5161⟩ },
  { kind := .ident "base", span := ⟨5161, 5165⟩ }]

def tableL1056 : List Token :=
  [{ kind := .comma, span := ⟨5165, 5166⟩ },
  { kind := .ident "slot", span := ⟨5167, 5171⟩ },
  { kind := .rparen, span := ⟨5171, 5172⟩ },
  { kind := .semi, span := ⟨5172, 5173⟩ },
  { kind := .newline, span := ⟨5173, 5174⟩ },
  { kind := .ident "oldVal", span := ⟨5176, 5182⟩ },
  { kind := .assign, span := ⟨5183, 5184⟩ },
  { kind := .ident "tableSlotVal", span := ⟨5185, 5197⟩ },
  { kind := .lparen, span := ⟨5197, 5198⟩ },
  { kind := .ident "base", span := ⟨5198, 5202⟩ },
  { kind := .comma, span := ⟨5202, 5203⟩ },
  { kind := .ident "slot", span := ⟨5204, 5208⟩ },
  { kind := .rparen, span := ⟨5208, 5209⟩ },
  { kind := .semi, span := ⟨5209, 5210⟩ },
  { kind := .newline, span := ⟨5210, 5211⟩ },
  { kind := .ident "if", span := ⟨5213, 5215⟩ },
  { kind := .ident "oldKey", span := ⟨5216, 5222⟩ },
  { kind := .ident "is", span := ⟨5223, 5225⟩ },
  { kind := .ident "TABLE_EMPTY", span := ⟨5226, 5237⟩ },
  { kind := .lbrace, span := ⟨5238, 5239⟩ },
  { kind := .newline, span := ⟨5239, 5240⟩ },
  { kind := .ident "newVal", span := ⟨5244, 5250⟩ },
  { kind := .assign, span := ⟨5251, 5252⟩ },
  { kind := .ident "TABLE_PRESENT", span := ⟨5253, 5266⟩ }]

def tableL1080 : List Token :=
  [{ kind := .ident "if", span := ⟨5267, 5269⟩ },
  { kind := .ident "merge", span := ⟨5270, 5275⟩ },
  { kind := .ident "is", span := ⟨5276, 5278⟩ },
  { kind := .ident "MERGE_SET", span := ⟨5279, 5288⟩ },
  { kind := .ident "else", span := ⟨5289, 5293⟩ },
  { kind := .ident "value", span := ⟨5294, 5299⟩ },
  { kind := .semi, span := ⟨5299, 5300⟩ },
  { kind := .newline, span := ⟨5300, 5301⟩ },
  { kind := .ident "tableSlotSet", span := ⟨5305, 5317⟩ },
  { kind := .lparen, span := ⟨5317, 5318⟩ },
  { kind := .ident "base", span := ⟨5318, 5322⟩ },
  { kind := .comma, span := ⟨5322, 5323⟩ },
  { kind := .ident "slot", span := ⟨5324, 5328⟩ },
  { kind := .comma, span := ⟨5328, 5329⟩ },
  { kind := .ident "key", span := ⟨5330, 5333⟩ },
  { kind := .comma, span := ⟨5333, 5334⟩ },
  { kind := .ident "newVal", span := ⟨5335, 5341⟩ },
  { kind := .rparen, span := ⟨5341, 5342⟩ },
  { kind := .semi, span := ⟨5342, 5343⟩ },
  { kind := .newline, span := ⟨5343, 5344⟩ },
  { kind := .ident "return", span := ⟨5348, 5354⟩ },
  { kind := .ident "newVal", span := ⟨5355, 5361⟩ },
  { kind := .semi, span := ⟨5361, 5362⟩ },
  { kind := .newline, span := ⟨5362, 5363⟩ }]

def tableL1104 : List Token :=
  [{ kind := .rbrace, span := ⟨5365, 5366⟩ },
  { kind := .newline, span := ⟨5366, 5367⟩ },
  { kind := .ident "newVal", span := ⟨5369, 5375⟩ },
  { kind := .assign, span := ⟨5376, 5377⟩ },
  { kind := .ident "TABLE_PRESENT", span := ⟨5378, 5391⟩ },
  { kind := .ident "if", span := ⟨5392, 5394⟩ },
  { kind := .ident "merge", span := ⟨5395, 5400⟩ },
  { kind := .ident "is", span := ⟨5401, 5403⟩ },
  { kind := .ident "MERGE_SET", span := ⟨5404, 5413⟩ },
  { kind := .ident "else", span := ⟨5414, 5418⟩ },
  { kind := .ident "value", span := ⟨5419, 5424⟩ },
  { kind := .semi, span := ⟨5424, 5425⟩ },
  { kind := .newline, span := ⟨5425, 5426⟩ },
  { kind := .ident "if", span := ⟨5428, 5430⟩ },
  { kind := .ident "merge", span := ⟨5431, 5436⟩ },
  { kind := .ident "is", span := ⟨5437, 5439⟩ },
  { kind := .ident "MERGE_ACC", span := ⟨5440, 5449⟩ },
  { kind := .lbrace, span := ⟨5450, 5451⟩ },
  { kind := .ident "newVal", span := ⟨5452, 5458⟩ },
  { kind := .assign, span := ⟨5459, 5460⟩ },
  { kind := .ident "oldVal", span := ⟨5461, 5467⟩ },
  { kind := .other (Char.ofNat 43), span := ⟨5468, 5469⟩ },
  { kind := .ident "value", span := ⟨5470, 5475⟩ },
  { kind := .semi, span := ⟨5475, 5476⟩ }]

def tableL1128 : List Token :=
  [{ kind := .rbrace, span := ⟨5477, 5478⟩ },
  { kind := .newline, span := ⟨5478, 5479⟩ },
  { kind := .ident "return", span := ⟨5481, 5487⟩ },
  { kind := .ident "tableSlotValSet", span := ⟨5488, 5503⟩ },
  { kind := .lparen, span := ⟨5503, 5504⟩ },
  { kind := .ident "base", span := ⟨5504, 5508⟩ },
  { kind := .comma, span := ⟨5508, 5509⟩ },
  { kind := .ident "slot", span := ⟨5510, 5514⟩ },
  { kind := .comma, span := ⟨5514, 5515⟩ },
  { kind := .ident "newVal", span := ⟨5516, 5522⟩ },
  { kind := .rparen, span := ⟨5522, 5523⟩ },
  { kind := .semi, span := ⟨5523, 5524⟩ },
  { kind := .newline, span := ⟨5524, 5525⟩ },
  { kind := .rbrace, span := ⟨5525, 5526⟩ },
  { kind := .newline, span := ⟨5526, 5527⟩ },
  { kind := .newline, span := ⟨5527, 5528⟩ },
  { kind := .newline, span := ⟨5528, 5529⟩ },
  { kind := .newline, span := ⟨5567, 5568⟩ },
  { kind := .ident "tableSet", span := ⟨5568, 5576⟩ },
  { kind := .lparen, span := ⟨5576, 5577⟩ },
  { kind := .ident "base", span := ⟨5577, 5581⟩ },
  { kind := .comma, span := ⟨5581, 5582⟩ },
  { kind := .ident "cap", span := ⟨5583, 5586⟩ },
  { kind := .comma, span := ⟨5586, 5587⟩ }]

def tableL1152 : List Token :=
  [{ kind := .ident "key", span := ⟨5588, 5591⟩ },
  { kind := .rparen, span := ⟨5591, 5592⟩ },
  { kind := .lbrace, span := ⟨5593, 5594⟩ },
  { kind := .newline, span := ⟨5594, 5595⟩ },
  { kind := .ident "return", span := ⟨5597, 5603⟩ },
  { kind := .ident "tableInsertWith", span := ⟨5604, 5619⟩ },
  { kind := .lparen, span := ⟨5619, 5620⟩ },
  { kind := .ident "base", span := ⟨5620, 5624⟩ },
  { kind := .comma, span := ⟨5624, 5625⟩ },
  { kind := .ident "cap", span := ⟨5626, 5629⟩ },
  { kind := .comma, span := ⟨5629, 5630⟩ },
  { kind := .ident "key", span := ⟨5631, 5634⟩ },
  { kind := .comma, span := ⟨5634, 5635⟩ },
  { kind := .ident "TABLE_PRESENT", span := ⟨5636, 5649⟩ },
  { kind := .comma, span := ⟨5649, 5650⟩ },
  { kind := .ident "MERGE_SET", span := ⟨5651, 5660⟩ },
  { kind := .rparen, span := ⟨5660, 5661⟩ },
  { kind := .semi, span := ⟨5661, 5662⟩ },
  { kind := .newline, span := ⟨5662, 5663⟩ },
  { kind := .rbrace, span := ⟨5663, 5664⟩ },
  { kind := .newline, span := ⟨5664, 5665⟩ },
  { kind := .newline, span := ⟨5665, 5666⟩ },
  { kind := .newline, span := ⟨5704, 5705⟩ },
  { kind := .ident "tablePut", span := ⟨5705, 5713⟩ }]

def tableL1176 : List Token :=
  [{ kind := .lparen, span := ⟨5713, 5714⟩ },
  { kind := .ident "base", span := ⟨5714, 5718⟩ },
  { kind := .comma, span := ⟨5718, 5719⟩ },
  { kind := .ident "cap", span := ⟨5720, 5723⟩ },
  { kind := .comma, span := ⟨5723, 5724⟩ },
  { kind := .ident "key", span := ⟨5725, 5728⟩ },
  { kind := .comma, span := ⟨5728, 5729⟩ },
  { kind := .ident "value", span := ⟨5730, 5735⟩ },
  { kind := .rparen, span := ⟨5735, 5736⟩ },
  { kind := .lbrace, span := ⟨5737, 5738⟩ },
  { kind := .newline, span := ⟨5738, 5739⟩ },
  { kind := .ident "return", span := ⟨5741, 5747⟩ },
  { kind := .ident "tableInsertWith", span := ⟨5748, 5763⟩ },
  { kind := .lparen, span := ⟨5763, 5764⟩ },
  { kind := .ident "base", span := ⟨5764, 5768⟩ },
  { kind := .comma, span := ⟨5768, 5769⟩ },
  { kind := .ident "cap", span := ⟨5770, 5773⟩ },
  { kind := .comma, span := ⟨5773, 5774⟩ },
  { kind := .ident "key", span := ⟨5775, 5778⟩ },
  { kind := .comma, span := ⟨5778, 5779⟩ },
  { kind := .ident "value", span := ⟨5780, 5785⟩ },
  { kind := .comma, span := ⟨5785, 5786⟩ },
  { kind := .ident "MERGE_MAP", span := ⟨5787, 5796⟩ },
  { kind := .rparen, span := ⟨5796, 5797⟩ }]

def tableL1200 : List Token :=
  [{ kind := .semi, span := ⟨5797, 5798⟩ },
  { kind := .newline, span := ⟨5798, 5799⟩ },
  { kind := .rbrace, span := ⟨5799, 5800⟩ },
  { kind := .newline, span := ⟨5800, 5801⟩ },
  { kind := .newline, span := ⟨5801, 5802⟩ },
  { kind := .newline, span := ⟨5836, 5837⟩ },
  { kind := .ident "tableSetPresent", span := ⟨5837, 5852⟩ },
  { kind := .lparen, span := ⟨5852, 5853⟩ },
  { kind := .ident "base", span := ⟨5853, 5857⟩ },
  { kind := .comma, span := ⟨5857, 5858⟩ },
  { kind := .ident "size", span := ⟨5859, 5863⟩ },
  { kind := .comma, span := ⟨5863, 5864⟩ },
  { kind := .ident "key", span := ⟨5865, 5868⟩ },
  { kind := .rparen, span := ⟨5868, 5869⟩ },
  { kind := .lbrace, span := ⟨5870, 5871⟩ },
  { kind := .newline, span := ⟨5871, 5872⟩ },
  { kind := .ident "return", span := ⟨5874, 5880⟩ },
  { kind := .ident "tableInsertWith", span := ⟨5881, 5896⟩ },
  { kind := .lparen, span := ⟨5896, 5897⟩ },
  { kind := .ident "base", span := ⟨5897, 5901⟩ },
  { kind := .comma, span := ⟨5901, 5902⟩ },
  { kind := .ident "size", span := ⟨5903, 5907⟩ },
  { kind := .comma, span := ⟨5907, 5908⟩ },
  { kind := .ident "key", span := ⟨5909, 5912⟩ }]

def tableL1224 : List Token :=
  [{ kind := .comma, span := ⟨5912, 5913⟩ },
  { kind := .ident "TABLE_PRESENT", span := ⟨5914, 5927⟩ },
  { kind := .comma, span := ⟨5927, 5928⟩ },
  { kind := .ident "MERGE_SET", span := ⟨5929, 5938⟩ },
  { kind := .rparen, span := ⟨5938, 5939⟩ },
  { kind := .semi, span := ⟨5939, 5940⟩ },
  { kind := .newline, span := ⟨5940, 5941⟩ },
  { kind := .rbrace, span := ⟨5941, 5942⟩ },
  { kind := .newline, span := ⟨5942, 5943⟩ },
  { kind := .newline, span := ⟨5943, 5944⟩ },
  { kind := .newline, span := ⟨5986, 5987⟩ },
  { kind := .ident "tableAdd", span := ⟨5987, 5995⟩ },
  { kind := .lparen, span := ⟨5995, 5996⟩ },
  { kind := .ident "base", span := ⟨5996, 6000⟩ },
  { kind := .comma, span := ⟨6000, 6001⟩ },
  { kind := .ident "size", span := ⟨6002, 6006⟩ },
  { kind := .comma, span := ⟨6006, 6007⟩ },
  { kind := .ident "key", span := ⟨6008, 6011⟩ },
  { kind := .comma, span := ⟨6011, 6012⟩ },
  { kind := .ident "value", span := ⟨6013, 6018⟩ },
  { kind := .rparen, span := ⟨6018, 6019⟩ },
  { kind := .lbrace, span := ⟨6020, 6021⟩ },
  { kind := .newline, span := ⟨6021, 6022⟩ },
  { kind := .ident "return", span := ⟨6024, 6030⟩ }]

def tableL1248 : List Token :=
  [{ kind := .ident "tableInsertWith", span := ⟨6031, 6046⟩ },
  { kind := .lparen, span := ⟨6046, 6047⟩ },
  { kind := .ident "base", span := ⟨6047, 6051⟩ },
  { kind := .comma, span := ⟨6051, 6052⟩ },
  { kind := .ident "size", span := ⟨6053, 6057⟩ },
  { kind := .comma, span := ⟨6057, 6058⟩ },
  { kind := .ident "key", span := ⟨6059, 6062⟩ },
  { kind := .comma, span := ⟨6062, 6063⟩ },
  { kind := .ident "value", span := ⟨6064, 6069⟩ },
  { kind := .comma, span := ⟨6069, 6070⟩ },
  { kind := .ident "MERGE_ACC", span := ⟨6071, 6080⟩ },
  { kind := .rparen, span := ⟨6080, 6081⟩ },
  { kind := .semi, span := ⟨6081, 6082⟩ },
  { kind := .newline, span := ⟨6082, 6083⟩ },
  { kind := .rbrace, span := ⟨6083, 6084⟩ },
  { kind := .newline, span := ⟨6084, 6085⟩ },
  { kind := .newline, span := ⟨6085, 6086⟩ },
  { kind := .newline, span := ⟨6176, 6177⟩ },
  { kind := .ident "mergeAcc", span := ⟨6177, 6185⟩ },
  { kind := .lparen, span := ⟨6185, 6186⟩ },
  { kind := .ident "left", span := ⟨6186, 6190⟩ },
  { kind := .comma, span := ⟨6190, 6191⟩ },
  { kind := .ident "right", span := ⟨6192, 6197⟩ },
  { kind := .rparen, span := ⟨6197, 6198⟩ }]

def tableL1272 : List Token :=
  [{ kind := .lbrace, span := ⟨6199, 6200⟩ },
  { kind := .ident "return", span := ⟨6201, 6207⟩ },
  { kind := .ident "left", span := ⟨6208, 6212⟩ },
  { kind := .other (Char.ofNat 43), span := ⟨6213, 6214⟩ },
  { kind := .ident "right", span := ⟨6215, 6220⟩ },
  { kind := .semi, span := ⟨6220, 6221⟩ },
  { kind := .rbrace, span := ⟨6222, 6223⟩ },
  { kind := .newline, span := ⟨6223, 6224⟩ },
  { kind := .newline, span := ⟨6296, 6297⟩ },
  { kind := .ident "law", span := ⟨6297, 6300⟩ },
  { kind := .ident "mergeacc_comm", span := ⟨6301, 6314⟩ },
  { kind := .ident "for", span := ⟨6315, 6318⟩ },
  { kind := .ident "mergeAcc", span := ⟨6319, 6327⟩ },
  { kind := .other (Char.ofNat 58), span := ⟨6327, 6328⟩ },
  { kind := .newline, span := ⟨6328, 6329⟩ },
  { kind := .ident "given", span := ⟨6331, 6336⟩ },
  { kind := .numeral 0, span := ⟨6337, 6338⟩ },
  { kind := .le, span := ⟨6339, 6341⟩ },
  { kind := .ident "left", span := ⟨6342, 6346⟩ },
  { kind := .le, span := ⟨6347, 6349⟩ },
  { kind := .ident "COORD_CEILING", span := ⟨6350, 6363⟩ },
  { kind := .newline, span := ⟨6363, 6364⟩ },
  { kind := .ident "given", span := ⟨6366, 6371⟩ },
  { kind := .numeral 0, span := ⟨6372, 6373⟩ }]

def tableL1296 : List Token :=
  [{ kind := .le, span := ⟨6374, 6376⟩ },
  { kind := .ident "right", span := ⟨6377, 6382⟩ },
  { kind := .le, span := ⟨6383, 6385⟩ },
  { kind := .ident "COORD_CEILING", span := ⟨6386, 6399⟩ },
  { kind := .newline, span := ⟨6399, 6400⟩ },
  { kind := .ident "ret", span := ⟨6402, 6405⟩ },
  { kind := .assign, span := ⟨6406, 6407⟩ },
  { kind := .ident "mergeAcc", span := ⟨6408, 6416⟩ },
  { kind := .lparen, span := ⟨6416, 6417⟩ },
  { kind := .ident "right", span := ⟨6417, 6422⟩ },
  { kind := .comma, span := ⟨6422, 6423⟩ },
  { kind := .ident "left", span := ⟨6424, 6428⟩ },
  { kind := .rparen, span := ⟨6428, 6429⟩ },
  { kind := .newline, span := ⟨6429, 6430⟩ },
  { kind := .ident "mergeAcc3L", span := ⟨6430, 6440⟩ },
  { kind := .lparen, span := ⟨6440, 6441⟩ },
  { kind := .ident "left", span := ⟨6441, 6445⟩ },
  { kind := .comma, span := ⟨6445, 6446⟩ },
  { kind := .ident "mid", span := ⟨6447, 6450⟩ },
  { kind := .comma, span := ⟨6450, 6451⟩ },
  { kind := .ident "right", span := ⟨6452, 6457⟩ },
  { kind := .rparen, span := ⟨6457, 6458⟩ },
  { kind := .lbrace, span := ⟨6459, 6460⟩ },
  { kind := .ident "return", span := ⟨6461, 6467⟩ }]

def tableL1320 : List Token :=
  [{ kind := .ident "mergeAcc", span := ⟨6468, 6476⟩ },
  { kind := .lparen, span := ⟨6476, 6477⟩ },
  { kind := .ident "mergeAcc", span := ⟨6477, 6485⟩ },
  { kind := .lparen, span := ⟨6485, 6486⟩ },
  { kind := .ident "left", span := ⟨6486, 6490⟩ },
  { kind := .comma, span := ⟨6490, 6491⟩ },
  { kind := .ident "mid", span := ⟨6492, 6495⟩ },
  { kind := .rparen, span := ⟨6495, 6496⟩ },
  { kind := .comma, span := ⟨6496, 6497⟩ },
  { kind := .ident "right", span := ⟨6498, 6503⟩ },
  { kind := .rparen, span := ⟨6503, 6504⟩ },
  { kind := .semi, span := ⟨6504, 6505⟩ },
  { kind := .rbrace, span := ⟨6506, 6507⟩ },
  { kind := .newline, span := ⟨6507, 6508⟩ },
  { kind := .newline, span := ⟨6581, 6582⟩ },
  { kind := .ident "law", span := ⟨6582, 6585⟩ },
  { kind := .ident "mergeacc_assoc", span := ⟨6586, 6600⟩ },
  { kind := .ident "for", span := ⟨6601, 6604⟩ },
  { kind := .ident "mergeAcc3L", span := ⟨6605, 6615⟩ },
  { kind := .other (Char.ofNat 58), span := ⟨6615, 6616⟩ },
  { kind := .newline, span := ⟨6616, 6617⟩ },
  { kind := .ident "given", span := ⟨6619, 6624⟩ },
  { kind := .numeral 0, span := ⟨6625, 6626⟩ },
  { kind := .le, span := ⟨6627, 6629⟩ }]

def tableL1344 : List Token :=
  [{ kind := .ident "left", span := ⟨6630, 6634⟩ },
  { kind := .le, span := ⟨6635, 6637⟩ },
  { kind := .ident "COORD_CEILING", span := ⟨6638, 6651⟩ },
  { kind := .newline, span := ⟨6651, 6652⟩ },
  { kind := .ident "given", span := ⟨6654, 6659⟩ },
  { kind := .numeral 0, span := ⟨6660, 6661⟩ },
  { kind := .le, span := ⟨6662, 6664⟩ },
  { kind := .ident "mid", span := ⟨6665, 6668⟩ },
  { kind := .le, span := ⟨6669, 6671⟩ },
  { kind := .ident "COORD_CEILING", span := ⟨6672, 6685⟩ },
  { kind := .newline, span := ⟨6685, 6686⟩ },
  { kind := .ident "given", span := ⟨6688, 6693⟩ },
  { kind := .numeral 0, span := ⟨6694, 6695⟩ },
  { kind := .le, span := ⟨6696, 6698⟩ },
  { kind := .ident "right", span := ⟨6699, 6704⟩ },
  { kind := .le, span := ⟨6705, 6707⟩ },
  { kind := .ident "COORD_CEILING", span := ⟨6708, 6721⟩ },
  { kind := .newline, span := ⟨6721, 6722⟩ },
  { kind := .ident "ret", span := ⟨6724, 6727⟩ },
  { kind := .assign, span := ⟨6728, 6729⟩ },
  { kind := .ident "mergeAcc", span := ⟨6730, 6738⟩ },
  { kind := .lparen, span := ⟨6738, 6739⟩ },
  { kind := .ident "left", span := ⟨6739, 6743⟩ },
  { kind := .comma, span := ⟨6743, 6744⟩ }]

def tableL1368 : List Token :=
  [{ kind := .ident "mergeAcc", span := ⟨6745, 6753⟩ },
  { kind := .lparen, span := ⟨6753, 6754⟩ },
  { kind := .ident "mid", span := ⟨6754, 6757⟩ },
  { kind := .comma, span := ⟨6757, 6758⟩ },
  { kind := .ident "right", span := ⟨6759, 6764⟩ },
  { kind := .rparen, span := ⟨6764, 6765⟩ },
  { kind := .rparen, span := ⟨6765, 6766⟩ },
  { kind := .newline, span := ⟨6766, 6767⟩ },
  { kind := .ident "test", span := ⟨6767, 6771⟩ },
  { kind := .ident "mergeacc_ground", span := ⟨6772, 6787⟩ },
  { kind := .lbrace, span := ⟨6788, 6789⟩ },
  { kind := .ident "mergeAcc", span := ⟨6790, 6798⟩ },
  { kind := .lparen, span := ⟨6798, 6799⟩ },
  { kind := .numeral 4, span := ⟨6799, 6800⟩ },
  { kind := .comma, span := ⟨6800, 6801⟩ },
  { kind := .numeral 9, span := ⟨6802, 6803⟩ },
  { kind := .rparen, span := ⟨6803, 6804⟩ },
  { kind := .assign, span := ⟨6805, 6806⟩ },
  { kind := .numeral 13, span := ⟨6807, 6809⟩ },
  { kind := .rbrace, span := ⟨6810, 6811⟩ },
  { kind := .newline, span := ⟨6811, 6812⟩ },
  { kind := .newline, span := ⟨6812, 6813⟩ },
  { kind := .ident "TABLE_TEST_ARENA", span := ⟨6813, 6829⟩ },
  { kind := .assign, span := ⟨6830, 6831⟩ }]

def tableL1392 : List Token :=
  [{ kind := .ident "cells", span := ⟨6832, 6837⟩ },
  { kind := .lparen, span := ⟨6837, 6838⟩ },
  { kind := .numeral 32, span := ⟨6838, 6840⟩ },
  { kind := .rparen, span := ⟨6840, 6841⟩ },
  { kind := .newline, span := ⟨6841, 6842⟩ },
  { kind := .newline, span := ⟨6842, 6843⟩ },
  { kind := .newline, span := ⟨6937, 6938⟩ },
  { kind := .ident "tableTestSetup", span := ⟨6938, 6952⟩ },
  { kind := .lparen, span := ⟨6952, 6953⟩ },
  { kind := .rparen, span := ⟨6953, 6954⟩ },
  { kind := .lbrace, span := ⟨6955, 6956⟩ },
  { kind := .newline, span := ⟨6956, 6957⟩ },
  { kind := .ident "tableInit", span := ⟨6959, 6968⟩ },
  { kind := .lparen, span := ⟨6968, 6969⟩ },
  { kind := .ident "TABLE_TEST_ARENA", span := ⟨6969, 6985⟩ },
  { kind := .comma, span := ⟨6985, 6986⟩ },
  { kind := .numeral 16, span := ⟨6987, 6989⟩ },
  { kind := .rparen, span := ⟨6989, 6990⟩ },
  { kind := .semi, span := ⟨6990, 6991⟩ },
  { kind := .newline, span := ⟨6991, 6992⟩ },
  { kind := .ident "tableSetPresent", span := ⟨6994, 7009⟩ },
  { kind := .lparen, span := ⟨7009, 7010⟩ },
  { kind := .ident "TABLE_TEST_ARENA", span := ⟨7010, 7026⟩ },
  { kind := .comma, span := ⟨7026, 7027⟩ }]

def tableL1416 : List Token :=
  [{ kind := .numeral 16, span := ⟨7028, 7030⟩ },
  { kind := .comma, span := ⟨7030, 7031⟩ },
  { kind := .numeral 3, span := ⟨7032, 7033⟩ },
  { kind := .rparen, span := ⟨7033, 7034⟩ },
  { kind := .semi, span := ⟨7034, 7035⟩ },
  { kind := .newline, span := ⟨7035, 7036⟩ },
  { kind := .ident "tableAdd", span := ⟨7038, 7046⟩ },
  { kind := .lparen, span := ⟨7046, 7047⟩ },
  { kind := .ident "TABLE_TEST_ARENA", span := ⟨7047, 7063⟩ },
  { kind := .comma, span := ⟨7063, 7064⟩ },
  { kind := .numeral 16, span := ⟨7065, 7067⟩ },
  { kind := .comma, span := ⟨7067, 7068⟩ },
  { kind := .numeral 3, span := ⟨7069, 7070⟩ },
  { kind := .comma, span := ⟨7070, 7071⟩ },
  { kind := .numeral 5, span := ⟨7072, 7073⟩ },
  { kind := .rparen, span := ⟨7073, 7074⟩ },
  { kind := .semi, span := ⟨7074, 7075⟩ },
  { kind := .newline, span := ⟨7075, 7076⟩ },
  { kind := .ident "return", span := ⟨7078, 7084⟩ },
  { kind := .ident "tableGet", span := ⟨7085, 7093⟩ },
  { kind := .lparen, span := ⟨7093, 7094⟩ },
  { kind := .ident "TABLE_TEST_ARENA", span := ⟨7094, 7110⟩ },
  { kind := .comma, span := ⟨7110, 7111⟩ },
  { kind := .numeral 16, span := ⟨7112, 7114⟩ }]

def tableL1440 : List Token :=
  [{ kind := .comma, span := ⟨7114, 7115⟩ },
  { kind := .numeral 3, span := ⟨7116, 7117⟩ },
  { kind := .rparen, span := ⟨7117, 7118⟩ },
  { kind := .semi, span := ⟨7118, 7119⟩ },
  { kind := .newline, span := ⟨7119, 7120⟩ },
  { kind := .rbrace, span := ⟨7120, 7121⟩ },
  { kind := .newline, span := ⟨7121, 7122⟩ },
  { kind := .ident "test", span := ⟨7122, 7126⟩ },
  { kind := .ident "table_set_add_read", span := ⟨7127, 7145⟩ },
  { kind := .lbrace, span := ⟨7146, 7147⟩ },
  { kind := .ident "tableTestSetup", span := ⟨7148, 7162⟩ },
  { kind := .lparen, span := ⟨7162, 7163⟩ },
  { kind := .rparen, span := ⟨7163, 7164⟩ },
  { kind := .assign, span := ⟨7165, 7166⟩ },
  { kind := .numeral 6, span := ⟨7167, 7168⟩ },
  { kind := .rbrace, span := ⟨7169, 7170⟩ },
  { kind := .newline, span := ⟨7170, 7171⟩ }]

def tableC1440 : List Token := tableL1440

def tableC1416 : List Token := tableL1416 ++ tableC1440

def tableC1392 : List Token := tableL1392 ++ tableC1416

def tableC1368 : List Token := tableL1368

def tableC1344 : List Token := tableL1344 ++ tableC1368

def tableC1320 : List Token := tableL1320

def tableC1296 : List Token := tableL1296 ++ tableC1320

def tableC1272 : List Token := tableL1272

def tableC1248 : List Token := tableL1248

def tableC1224 : List Token := tableL1224 ++ tableC1248

def tableC1200 : List Token := tableL1200 ++ tableC1224

def tableC1176 : List Token := tableL1176 ++ tableC1200

def tableC1152 : List Token := tableL1152

def tableC1128 : List Token := tableL1128 ++ tableC1152

def tableC1104 : List Token := tableL1104 ++ tableC1128

def tableC1080 : List Token := tableL1080 ++ tableC1104

def tableC1056 : List Token := tableL1056 ++ tableC1080

def tableC1032 : List Token := tableL1032 ++ tableC1056

def tableC1008 : List Token := tableL1008 ++ tableC1032

def tableC984 : List Token := tableL984 ++ tableC1008

def tableC960 : List Token := tableL960 ++ tableC984

def tableC936 : List Token := tableL936 ++ tableC960

def tableC912 : List Token := tableL912 ++ tableC936

def tableC888 : List Token := tableL888

def tableC864 : List Token := tableL864 ++ tableC888

def tableC840 : List Token := tableL840 ++ tableC864

def tableC816 : List Token := tableL816 ++ tableC840

def tableC792 : List Token := tableL792 ++ tableC816

def tableC768 : List Token := tableL768 ++ tableC792

def tableC744 : List Token := tableL744 ++ tableC768

def tableC720 : List Token := tableL720 ++ tableC744

def tableC696 : List Token := tableL696

def tableC672 : List Token := tableL672

def tableC648 : List Token := tableL648 ++ tableC672

def tableC624 : List Token := tableL624 ++ tableC648

def tableC600 : List Token := tableL600 ++ tableC624

def tableC576 : List Token := tableL576

def tableC552 : List Token := tableL552

def tableC528 : List Token := tableL528

def tableC504 : List Token := tableL504 ++ tableC528

def tableC480 : List Token := tableL480 ++ tableC504

def tableC456 : List Token := tableL456 ++ tableC480

def tableC432 : List Token := tableL432 ++ tableC456

def tableC408 : List Token := tableL408 ++ tableC432

def tableC384 : List Token := tableL384

def tableC360 : List Token := tableL360

def tableC336 : List Token := tableL336 ++ tableC360

def tableC312 : List Token := tableL312

def tableC288 : List Token := tableL288

def tableC264 : List Token := tableL264 ++ tableC288

def tableC240 : List Token := tableL240

def tableC216 : List Token := tableL216

def tableC192 : List Token := tableL192 ++ tableC216

def tableC168 : List Token := tableL168

def tableC144 : List Token := tableL144

def tableC120 : List Token := tableL120 ++ tableC144

def tableC96 : List Token := tableL96

def tableC72 : List Token := tableL72

def tableC48 : List Token := tableL48 ++ tableC72

def tableC24 : List Token := tableL24

def tableC0 : List Token := tableL0

end Slang.Generated.SourceWitness.Decoder
