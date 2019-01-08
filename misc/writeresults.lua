
-- adapted from https://github.com/karpathy/char-rnn

require 'torch'
local utils = {}

function utils.clone_list(tensor_list, zero_too)
    -- takes a list of tensors and returns a list of cloned tensors
    local out = {}
    for k,v in pairs(tensor_list) do
        out[k] = v:clone()
        if zero_too then out[k]:zero() end
    end
    return out
end

function utils.calc_accuracy(res)
  local gt, predict, length = res.gt, res.predict, res.length
  local v_gt, v_predict = res.v_gt, res.v_predict
  local num_data = length:nElement()
  local verb_predict = torch.zeros(num_data)
  local noun_predict = torch.zeros(num_data, 6)
  local logp         = torch.zeros(num_data)
  print('#samples: ', num_data)

  local agents_idx, places_idx = 2, 1
  local agents_1  = torch.ByteTensor(num_data):fill(0)
  local places_1  = torch.ByteTensor(num_data):fill(0)
  local value     = torch.IntTensor(num_data):fill(0)
  local value_num = torch.IntTensor(num_data):fill(0)
  local value_any  = torch.ByteTensor(num_data):fill(1)
  local value_full = torch.ByteTensor(num_data):fill(0)
  local verbs_1   = torch.ByteTensor(num_data):fill(0)
  local verbs_5   = torch.ByteTensor(num_data):fill(0)
  local none_agents = 0
  local none_places = 0
  for i = 1, num_data do

    -- verb prediction
    local v_gt_i, v_predict_i = v_gt[i], v_predict[i]
    if v_gt_i == v_predict_i[1] then verbs_1[i] = 1 end
    if torch.any( torch.eq(v_predict_i, v_gt_i) ) then
      verbs_5[i] = 1
    end

    -- nouns prediction
    local gt_i, predict_i = gt[i], predict[i]

    -- record prediction
    verb_predict[i] = v_predict_i[1]
    for j = 1, 6 do
      noun_predict[i][j] = predict_i[j]
    end
    logp[i] = res.logp[i]

    -- agents
    local I = torch.eq(gt_i[{{},{agents_idx}}], predict_i[agents_idx])
    if torch.any(I) then agents_1[i] = 1 end

    -- places
    local J = torch.eq(gt_i[{{},{places_idx}}], predict_i[places_idx])
    if torch.any(J) then places_1[i] = 1 end

    -- value & value_any
    value_num[i] = length[i]
    if gt_i[1][agents_idx] == 0 then -- some verbs don't have agent
      value_any[i] = places_1[i]
      none_agents = none_agents + 1
      value_num[i] = value_num[i] - 1
    elseif gt_i[1][places_idx] == 0 then -- some verbs don't have place
      value_any[i] = agents_1[i]
      none_places = none_places + 1
      value_num[i] = value_num[i] - 1
    else
      if (agents_1[i] == 0 or places_1[i] == 0) then
        value_any[i] = 0
      end
    end

    value[i] = agents_1[i] + places_1[i]
    for j = 3, length[i] do
      local K = torch.eq(gt_i[{{},{j}}], predict_i[j])
      if not torch.any(K) then 
        value_any[i] = 0
      else
        value[i] = value[i] + 1
      end
    end

    -- value_full
    local K = torch.ByteTensor(3):fill(1)
    for j = 1, length[i] do
      if gt_i[1][j] == 0 then
        -- some verbs don't have agent or place
      else
        if not (predict_i[j] == gt_i[1][j]) then K[1] = 0 end
        if not (predict_i[j] == gt_i[2][j]) then K[2] = 0 end
        if not (predict_i[j] == gt_i[3][j]) then K[3] = 0 end
      end
    end
    if torch.any(K) then value_full[i] = 1 end

  end

  Acc = {}
  Acc.agents_top1 = torch.sum(agents_1) / (num_data - none_agents)
  Acc.places_top1 = torch.sum(places_1) / (num_data - none_places)
  Acc.value       = torch.sum(value) / torch.sum(value_num)
  Acc.value_any   = torch.sum(value_any) / num_data
  Acc.value_full  = torch.sum(value_full) / num_data

  local I = torch.eq(verbs_1, 1)
  Acc.verbs_top1        = torch.sum(verbs_1) / num_data
  Acc.value_top1        = torch.sum(value[I]) / torch.sum(value_num)
  Acc.value_any_top1    = torch.sum(value_any[I]) / num_data
  Acc.value_full_top1   = torch.sum(value_full[I]) / num_data

  local J = torch.eq(verbs_5, 1)
  Acc.verbs_top5        = torch.sum(verbs_5) / num_data
  Acc.value_top5        = torch.sum(value[J]) / torch.sum(value_num)
  Acc.value_any_top5    = torch.sum(value_any[J]) / num_data
  Acc.value_full_top5   = torch.sum(value_full[J]) / num_data

  local hdf5 = require 'hdf5'
  local fp = hdf5.open('./value_verb.h5', 'w')
  fp:write('/value', value);
  -- fp:write('/value_num', value_num);
  fp:write('/value_any', value_any);
  fp:write('/value_full', value_full);
  fp:write('/verbs_1', verbs_1);
  fp:write('/verbs_5', verbs_5);
  fp:write('/verb_predict', verb_predict);
  fp:write('/noun_predict', noun_predict);
  fp:write('/logp', logp);
  fp:close()

  return Acc
end

return utils