require 'model.LinearNB'

local GatedGraph = {}

--[[
Creates one timestep of one gated graph

There are (T+N)*2+1 inputs and (T+N) outputs,
first (T+N) inputs are the input features (size: BxD),
next (T+N) inputs encode the structure (size: Bx(T+N)),
last input represents the mask (size: Bx(T+N)).

Note: this unit outputs raw Vector instead of LogSoftMax.
]]--

function GatedGraph.graph(rnn_size, T, N, dropout)
  -- dropout = dropout or 0
  dropout = 0 -- we don't use dropout in ggnn
  local nInputs = T + N

  function new_input(xx, insize, outsize, NNN)
    NNN = NNN or nInputs
    local m = nn.MapTable():add(nn.LinearNB(insize, outsize))
    return { m(xx):split(NNN) }
  end

  function mask_state(states, mm)
    local states_masked = {}
    for n = 1, nInputs do
      states_masked[n] = nn.CMulTable()({states[n], mm[n]})
    end
    return states_masked
  end

  -- inputs
  local inputs = {}
  for n = 1, nInputs*2+1 do
    table.insert(inputs, nn.Identity()())
  end

  -- mask the hidden states of nonexistent nodes
  local mask = inputs[#inputs]
  local split_mask = { nn.SplitTable(2)(mask):split(nInputs) } -- split mask along the 2nd dimenstion
  local expand_mask = {}
  for n = 1, nInputs do
    expand_mask[n] = nn.Replicate(rnn_size, 2)(split_mask[n])
  end
  local hidden_init = {}
  for n = 1, nInputs do
    hidden_init[n] = inputs[n]
  end
  local hidden = mask_state(hidden_init, expand_mask)

  -- message passing
  local shared_nodes = {}
  for n = T+1, T+N do
    table.insert(shared_nodes, hidden[n])
  end
  local msg = {}
  local msg_shared = {}
  local msg_node = {}
  for i = 1, T + 1 do
    msg[i] = {}
    for j = 1, T do
      table.insert(msg[i], nn.LinearNB(rnn_size, rnn_size)(hidden[j]))
    end
    msg_shared[i] = new_input(shared_nodes, rnn_size, rnn_size, N)
    for j = 1, N do
      table.insert(msg[i], msg_shared[i][j])
    end
    msg_node[i] = nn.NarrowTable(1, nInputs)(msg[i])
  end
  local activs = {}
  for n = 1, nInputs do
    if n <= T then
      activs[n] = nn.MixtureTable(2){inputs[n+nInputs], msg_node[n]}
    else
      activs[n] = nn.MixtureTable(2){inputs[n+nInputs], msg_node[T+1]}
    end
  end

  -- forward the update and reset gates
  local Wz_a = new_input(activs, rnn_size, rnn_size)
  local Uz_h = new_input(hidden, rnn_size, rnn_size)
  local Wr_a = new_input(activs, rnn_size, rnn_size)
  local Ur_h = new_input(hidden, rnn_size, rnn_size)
  local update_gate = {}
  local reset_gate = {}
  for n = 1, nInputs do
    update_gate[n] = nn.Sigmoid()(nn.CAddTable()({Wz_a[n], Uz_h[n]}))
    reset_gate[n] = nn.Sigmoid()(nn.CAddTable()({Wr_a[n], Ur_h[n]}))
  end

  -- compute candidate hidden state
  local gated_hidden = {}
  for n = 1, nInputs do
    gated_hidden[n] = nn.CMulTable()({reset_gate[n], hidden[n]})
  end
  local W_a = new_input(activs, rnn_size, rnn_size)
  local U_h = new_input(gated_hidden, rnn_size, rnn_size)
  local hidden_candidate = {}
  for n = 1, nInputs do
    hidden_candidate[n] = nn.Tanh()(nn.CAddTable()({W_a[n], U_h[n]}))
  end

  -- compute new interpolated hidden state, based on the update gate
  local outputs = {}
  for n = 1, nInputs do
    local zh = nn.CMulTable()({update_gate[n], hidden_candidate[n]})
    local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate[n])), hidden[n]})
    local zh_sum = nn.CAddTable()({zh, zhm1})
    if (dropout > 0) then zh_sum = nn.Dropout(dropout)(zh_sum) end
    table.insert(outputs, zh_sum)
  end

  -- mask the output states
  local outputs_masked = mask_state(outputs, expand_mask)

  return nn.gModule(inputs, outputs_masked)
end

return GatedGraph
