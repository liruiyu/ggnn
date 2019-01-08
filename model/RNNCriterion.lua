require 'nn'

-- input comes from nn.LogSoftMax()
-- label = 0 indicates loss won't be computed

local crit, parent = torch.class('nn.RNNCriterion', 'nn.Criterion')
function crit:__init()
  parent.__init(self)
end

function crit:updateOutput(input, label)
  self.gradInput:resizeAs(input):zero() -- reset to zeros
  local batch_size = input:size(1)

  local loss = 0
  local n = 0
  for b = 1, batch_size do
    local target_index = label[b]
    if target_index ~= 0 then
      loss = loss - input[{ b, target_index }]
      self.gradInput[{ b, target_index }] = -1
      n = n + 1
    end
  end

  self.output = loss / (n + 10e-8)
  self.gradInput:div((n + 10e-8))
  return self.output
end

function crit:updateGradInput(input, label)
  return self.gradInput
end