local BinarySpatialConvolution, parent = torch.class('BinarySpatialConvolution', 'nn.SpatialConvolution')

function BinarySpatialConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
  local delayedReset = self.reset
  self.reset = function() end
  parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH)
  self.reset = delayedReset
  self.padW = padW or 0
  self.padH = padH or 0
  self.stcWeights = stcWeights or false
  self.groups = groups or 1
  assert(nInputPlane % self.groups == 0,
         'nInputPlane should be divisible by nGroups')
  assert(nOutputPlane % self.groups == 0,
         'nOutputPlane should be divisible by nGroups')
  self.weight = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kW, kH)
  self.weightB = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kW, kH)
  self.weightOrg = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kW, kH)
  self.randmat = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kW, kH)
  self.maskStc = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kW, kH)
  self:reset()
  -- should nil for serialization, the reset will still work
  self.reset = nil
  self.iSize = torch.LongStorage(4):fill(0)


end

function BinarySpatialConvolution:reset(stdv)
  if stdv then
     stdv = stdv * math.sqrt(3)
  else
     stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
  end
  if nn.oldSeed then
     self.weight:apply(function()
        return torch.uniform(-1, 1)
     end)
     if self.bias then
        self.bias:apply(function()
        return torch.uniform(-stdv, stdv)
        end)
     end
  else
     self.weight:uniform(-1, 1)
     if self.bias then
        self.bias:uniform(-stdv, stdv)
     end
  end
end

function BinarySpatialConvolution:binarized(trainFlag)
  self.weightOrg:copy(self.weight)
  self.binaryFlag = true
  if not self.binaryFlag then
    self.weight:copy(self.weightOrg)
  else
    self.weightB:copy(self.weight):add(1):div(2):clamp(0,1)

    if not self.stcWeights or not trainFlag then
      self.weightB:round():mul(2):add(-1)
    else
      self.maskStc=self.weightB-self.randmat:rand(self.randmat:size())
      self.weightB:copy(self.maskStc)

    end
  end

  return  self.weightB
end

local function backCompatibility(self)
   self.finput = self.finput or self.weight.new()
   self.fgradInput = self.fgradInput or self.weight.new()
   if self.padding then
      self.padW = self.padding
      self.padH = self.padding
      self.padding = nil
   else
      self.padW = self.padW or 0
      self.padH = self.padH or 0
   end
   if self.weight:dim() == 2 then
      self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   end
   if self.gradWeight and self.gradWeight:dim() == 2 then
      self.gradWeight = self.gradWeight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   end
end

local function makeContiguous(self, input, gradOutput)
  if not input:isContiguous() then
    self._input = self._input or input.new()
    self._input:resizeAs(input):copy(input)
    input = self._input
 end
 if gradOutput then
    if not gradOutput:isContiguous() then
 self._gradOutput = self._gradOutput or gradOutput.new()
 self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
 gradOutput = self._gradOutput
    end
 end
 return input, gradOutput
end

-- function to re-view the weight layout in a way that would make the MM ops happy
local function viewWeight(self)
   self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane * self.kH * self.kW)
   if self.gradWeight and self.gradWeight:dim() > 0 then
      self.gradWeight = self.gradWeight:view(self.nOutputPlane, self.nInputPlane * self.kH * self.kW)
   end
end

local function unviewWeight(self)
   self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   if self.gradWeight and self.gradWeight:dim() > 0 then
      self.gradWeight = self.gradWeight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   end
end

function BinarySpatialConvolution:updateOutput(input)
   backCompatibility(self)
   viewWeight(self)
   input = makeContiguous(self, input)
   self.weightB = self:binarized(self.train)
   self.weight:copy(self.weightB)
   input.THNN.SpatialConvolutionMM_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.weight:cdata(),
      self.bias:cdata(),
      self.finput:cdata(),
      self.fgradInput:cdata(),
      self.kW, self.kH,
      self.dW, self.dH,
      self.padW, self.padH
   )
   self.weight:copy(self.weightOrg)
   unviewWeight(self)
   return self.output
end

function BinarySpatialConvolution:updateGradInput(input, gradOutput)
   if self.gradInput then
      backCompatibility(self)
      viewWeight(self)
      input, gradOutput = makeContiguous(self, input, gradOutput)
      self.weight:copy(self.weightB)
      input.THNN.SpatialConvolutionMM_updateGradInput(
         input:cdata(),
         gradOutput:cdata(),
         self.gradInput:cdata(),
         self.weight:cdata(),
         -- self.bias:cdata(), -- removed from this commit https://github.com/torch/nn/commit/651103f3aabc2dd154d6bd95ad565d14009255e6
         self.finput:cdata(),
         self.fgradInput:cdata(),
         self.kW, self.kH,
         self.dW, self.dH,
         self.padW, self.padH
      )
      self.weight:copy(self.weightOrg)
      unviewWeight(self)
      return self.gradInput
   end
end

function BinarySpatialConvolution:accGradParameters(input, gradOutput, scale)
  scale = scale or 1
  backCompatibility(self)
  input, gradOutput = makeContiguous(self, input, gradOutput)
  viewWeight(self)
  input.THNN.SpatialConvolutionMM_accGradParameters(
     input:cdata(),
     gradOutput:cdata(),
     self.gradWeight:cdata(),
     self.gradBias:cdata(),
     self.finput:cdata(),
     self.fgradInput:cdata(),
     self.kW, self.kH,
     self.dW, self.dH,
     self.padW, self.padH,
     scale
  )
  unviewWeight(self)
end

function BinarySpatialConvolution:type(type,tensorCache)
   self.finput = self.finput and torch.Tensor()
   self.fgradInput = self.fgradInput and torch.Tensor()
   return parent.type(self,type,tensorCache)
end

function BinarySpatialConvolution:__tostring__()
   return parent.__tostring__(self)
end

function BinarySpatialConvolution:clearState()
   nn.utils.clear(self, 'finput', 'fgradInput', '_input', '_gradOutput')
   return parent.clearState(self)
end
