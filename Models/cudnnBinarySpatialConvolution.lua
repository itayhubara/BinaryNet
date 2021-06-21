local cudnnBinarySpatialConvolution, parent =
    torch.class('cudnnBinarySpatialConvolution', 'cudnn.SpatialConvolution')
local ffi = require 'ffi'
local errcheck = cudnn.errcheck

local autotunerCache = {}
autotunerCache[1] = {} -- forward
autotunerCache[2] = {} -- backwardFilter
autotunerCache[3] = {} -- backwardData

function cudnnBinarySpatialConvolution:__init(nInputPlane, nOutputPlane,
                            kW, kH, dW, dH, padW, padH,stcWeights, groups)
    local delayedReset = self.reset
    self.reset = function() end
    parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH)
    self.reset = delayedReset
    self.padW = padW or 0
    self.padH = padH or 0
    self.groups = groups or 1
    self.stcWeights = stcWeights or false
    assert(nInputPlane % self.groups == 0,
           'nInputPlane should be divisible by nGroups')
    assert(nOutputPlane % self.groups == 0,
           'nOutputPlane should be divisible by nGroups')
    self.weight = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kH, kW)
    self.weightB = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kW, kH)
    self.weightOrg = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kW, kH)
    self.randmat = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kW, kH)
    self.maskStc = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kW, kH)
    self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kH, kW)
    self:reset()
    -- should nil for serialization, the reset will still work
    self.reset = nil
end

function cudnnBinarySpatialConvolution:binarized(trainFlag)
  self.weightOrg:copy(self.weight)
  self.binaryFlag = true
  if not self.binaryFlag then
    self.weight:copy(self.weightOrg)
  else
    self.weightB:copy(self.weight):add(1):div(2):clamp(0,1)

    if not self.stcWeights or not trainFlag then
      self.weightB:round():mul(2):add(-1)
      --print(self.weightB)
    else
      self.maskStc=self.weightB-self.randmat:rand(self.randmat:size())
      self.weightB:copy(self.maskStc)

    end
  end

  return  self.weightB
end

-- if you change the configuration of the module manually, call this
function cudnnBinarySpatialConvolution:resetWeightDescriptors()
    assert(torch.typename(self.weight) == 'torch.CudaTensor',
           'Only Cuda supported duh!')
    assert(torch.typename(self.bias) == 'torch.CudaTensor' or not self.bias,
           'Only Cuda supported duh!')
    -- for compatibility
    self.groups = self.groups or 1
    -- create filterDescriptor for weight
    self.weightDesc = ffi.new('struct cudnnFilterStruct*[1]')
    errcheck('cudnnCreateFilterDescriptor', self.weightDesc)
    local desc = torch.IntTensor({self.nOutputPlane/self.groups,
                              self.nInputPlane/self.groups,
                              self.kH, self.kW})
    errcheck('cudnnSetFilterNdDescriptor', self.weightDesc[0],
             'CUDNN_DATA_FLOAT', 'CUDNN_TENSOR_NCHW', 4,
             desc:data());
    local function destroyWDesc(d)
        errcheck('cudnnDestroyFilterDescriptor', d[0]);
    end
    ffi.gc(self.weightDesc, destroyWDesc)

    -- create descriptor for bias
    if self.bias then
        self.biasDesc = cudnn.toDescriptor(self.bias:view(1, self.nOutputPlane,1,1))
    end
end

function cudnnBinarySpatialConvolution:fastest(mode)
    if mode == nil then mode = true end
    self.fastest_mode = mode
    self.iSize = self.iSize or torch.LongStorage(4)
    self.iSize:fill(0)
    return self
end

function cudnnBinarySpatialConvolution:setMode(fmode, bdmode, bwmode)
    if fmode ~= nil then
        self.fmode = fmode
    end
    if bdmode ~= nil then
        self.bdmode = bdmode
    end
    if bwmode ~= nil then
        self.bwmode = bwmode
    end
    self.iSize = self.iSize or torch.LongStorage(4)
    self.iSize:fill(0)
    return self
end

function cudnnBinarySpatialConvolution:resetMode()
    self.fmode = nil
    self.bdmode = nil
    self.bwmode = nil
    return self
end

function cudnnBinarySpatialConvolution:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end

function cudnnBinarySpatialConvolution:createIODescriptors(input)
    parent.createIODescriptors(self,input)
end

local one = torch.FloatTensor({1});
local zero = torch.FloatTensor({0});

local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:typeAs(input):resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput and not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new()
      self._gradOutput:typeAs(gradOutput):resizeAs(gradOutput):copy(gradOutput)
      gradOutput = self._gradOutput
   end
   return input, gradOutput
end

function cudnnBinarySpatialConvolution:updateOutput(input)
    self.weightOrg:copy(self.weight)
    self.weightB = self:binarized(self.train)
    self.weight:copy(self.weightB)
    parent.updateOutput(self,input)
    self.weight:copy(self.weightOrg)
    return self.output
end

function cudnnBinarySpatialConvolution:updateGradInput(input, gradOutput)
    if not self.gradInput then return end
    self.weight:copy(self.weightB)
    parent.updateGradInput(self, input, gradOutput:contiguous(), scale)
    self.weight:copy(self.weightOrg)
    return self.gradInput
end

function cudnnBinarySpatialConvolution:accGradParameters(input, gradOutput, scale)
    parent.accGradParameters(self, input, gradOutput:contiguous(), scale)
end

function cudnnBinarySpatialConvolution:clearDesc()
    self.weightDesc = nil
    self.biasDesc = nil
    self.convDesc = nil
    self.iDesc = nil
    self.oDesc = nil
    self.oDescForBias = nil
    self.algType = nil
    self.fwdAlgType = nil
    self.bwdDataAlgType = nil
    self.bwdFilterAlgType = nil
    self.extraBuffer = nil
    self.extraBufferSizeInBytes = nil
    self.scaleT = nil
end

function cudnnBinarySpatialConvolution:write(f)
    self:clearDesc()
    local var = {}
    for k,v in pairs(self) do
        var[k] = v
    end
    f:writeObject(var)
end

function cudnnBinarySpatialConvolution:clearState()
   self:clearDesc()
   return nn.Module.clearState(self)
end
