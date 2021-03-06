------------------------------------------------------------------------
--[[ BRNN ]] --
-- Encapsulates a forward, backward and merge module.
-- Input is a tensor e.g batch x time x inputdim.
-- Output is a tensor of the same length e.g batch x time x outputdim.
-- Applies a forward rnn to input tensor in forward order
-- and applies a backward rnn in reverse order.
-- Reversal of the sequence happens on the chosen dim (defaults to 2).
-- For each step, the outputs of both rnn are merged together using
-- the merge module (defaults to nn.CAddTable() which sums the activations).
--
-- copied from torch-rnn.
------------------------------------------------------------------------
require 'model/MS_ReverseSequence'


local MS_BRNN, parent = torch.class('nn.MS_BRNN', 'nn.Container')

function MS_BRNN:__init(forward, backward, merge, dimToReverse)
    if not torch.isTypeOf(forward, 'nn.Module') then
        error "MS_BRNN: expecting nn.Module instance at arg 1"
    end
    self.forwardModule = forward
    self.backwardModule = backward
    self.merge = merge
    self.dim = dimToReverse
    if not self.backwardModule then
        self.backwardModule = forward:clone()
        self.backwardModule:reset()
    end
    if not torch.isTypeOf(self.backwardModule, 'nn.Module') then
        error "MS_BRNN: expecting nn.Module instance at arg 2"
    end
    if not self.merge then
        self.merge = nn.CAddTable()
    end
    if not self.dim then
        self.dim = 2 -- default to second dimension to reverse (expecting batch x time x inputdim).
    end
    local backward = nn.Sequential()
    backward:add(nn.MS_ReverseSequence(self.dim)) -- reverse
    backward:add(self.backwardModule)
    backward:add(nn.MS_ReverseSequence(self.dim)) -- unreverse

    local concat = nn.ConcatTable()
    concat:add(self.forwardModule):add(backward)

    local brnn = nn.Sequential()
    brnn:add(concat)
    brnn:add(self.merge)

    parent.__init(self)

    self.output = torch.Tensor()
    self.gradInput = torch.Tensor()

    self.module = brnn
    -- so that it can be handled like a Container
    self.modules[1] = brnn
end

function MS_BRNN:updateType(dtype)
  self.module = self.module:type(dtype)
end

function MS_BRNN:updateOutput(input)
    self.output = self.module:updateOutput(input)
    return self.output
end

function MS_BRNN:updateGradInput(input, gradOutput)
    self.gradInput = self.module:updateGradInput(input, gradOutput)
    return self.gradInput
end

function MS_BRNN:accGradParameters(input, gradOutput, scale)
    self.module:accGradParameters(input, gradOutput, scale)
end

function MS_BRNN:accUpdateGradParameters(input, gradOutput, lr)
    self.module:accUpdateGradParameters(input, gradOutput, lr)
end

function MS_BRNN:sharedAccUpdateGradParameters(input, gradOutput, lr)
    self.module:sharedAccUpdateGradParameters(input, gradOutput, lr)
end

function MS_BRNN:resetStates()
  self.forwardModule:resetStates()
  self.backwardModule:resetStates()
end

function MS_BRNN:__tostring__()
    if self.module.__tostring__ then
        return torch.type(self) .. ' @ ' .. self.module:__tostring__()
    else
        return torch.type(self) .. ' @ ' .. torch.type(self.module)
    end
end

