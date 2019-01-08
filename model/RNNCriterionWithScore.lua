require 'nn'

-- input[1] comes from nn.LogSoftMax()
-- input[2]==1 indicates loss exists
-- label = 0 indicates loss won't be computed

local crit, parent = torch.class('nn.RNNCriterionWithScore', 'nn.Criterion')
function crit:__init()
  parent.__init(self)
end

function crit:updateOutput(input, label)
  self.gradInput:resizeAs(input[1]):zero() -- reset to zeros
  local batch_size = input[1]:size(1)

  local loss = 0
  local n = 0
  for b = 1, batch_size do
    local target_index = label[b]
    if input[2][b] == 1 and target_index ~= 0 then
      loss = loss - input[1][{ b, target_index }]
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