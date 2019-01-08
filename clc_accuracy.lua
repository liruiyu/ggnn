require 'torch'
require 'misc.DataLoader'
local utils = require 'misc.writeresults'
local color = require 'trepl.colorize'
torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()
cmd:option('-data', '')
cmd:option('-model', 'e4', 'e4|e5|e6|e7')
cmd:option('-beam_size', '10')
cmd:option('-alpha', '1')
local opt = cmd:parse(arg)
local beam_size = opt.beam_size
if type(beam_size) == 'string' then beam_size = tonumber(beam_size) end
local alpha = opt.alpha
if type(alpha) == 'string' then alpha = tonumber(alpha) end

local h5_file = '/mnt/sdd/ryli/situation/dataset/rnn_imgs_nouns.h5'
local save_t7_prefix = '/mnt/sdd/ryli/situation/results/' .. opt.data
-- local save_t7_prefix = '/mnt/sdd/ryli/situation/results/ronly/' .. opt.data
local num_files = 99

local loader = DataLoader{h5_file=h5_file, wo_preload=true}
local graph2verb, verb2graph  = loader:getConversions()
local graph                   = loader:getGraphs()
local verbs2roles             = loader:getVerbs2Roles()
local role_length             = torch.zeros(verbs2roles:size(1))
for i = 1, role_length:nElement() do
  role_length[i] = torch.sum( torch.ne(verbs2roles[{{i}}], 0) )
end


print('loading info ... ' .. save_t7_prefix)
local info = torch.load(save_t7_prefix .. '_info.t7')
local num_data = info.length:nElement()

print('loading data ...')
local v_predict = {}
local l_predict = {}
local sum_logp = {}
local k_data = 0

local graph_acc = {}
local graph_num = {}
for i = 1, #graph2verb do
  graph_num[i] = 0
  graph_acc[i] = {}
  for j = 1, #graph2verb do
    graph_acc[i][j] = 0 -- gt is i but predict j
  end
end

local function compare(a, b) return a.p > b.p end
local function sameGraph(a, b) return verb2graph[a] == verb2graph[b] end

function load_result_roleonly( filename )
  local data = torch.load(filename)
  for k = 1, #data.verb_prop do
    v_predict[k + k_data] = torch.Tensor(5)

    local PP, II = torch.sort(data.verb_prop[k]:view(-1), true)

    -- print('start beam search')
    if beam_size > 1 then
      local total_prop = torch.zeros(beam_size)
      for i = 1, beam_size do
        total_prop[i] = PP[i]
        local tmp_prop = 0
        for j = 1, role_length[ II[i] ] do
          tmp_prop = tmp_prop + data.noun_prop[k][ II[i] ][j]
        end
        -- print(total_prop[i] .. ' ' .. tmp_prop / role_length[ II[i] ])
        total_prop[i] = alpha * total_prop[i] + tmp_prop / role_length[ II[i] ]
      end
      QQ, JJ = torch.sort(total_prop, true)
    end

    for v = 1, 5 do
      if beam_size > 1 then
        v_predict[k + k_data][v] = II[ JJ[v] ]
        sum_logp[k + k_data] = QQ[1]
      else
        v_predict[k + k_data][v] = II[v]
        sum_logp[k + k_data] = PP[1]
      end
    end
  end

  return #data.verb_prop
end

function load_result_prgraph( filename )
  local data = torch.load(filename)
  for k = 1, #data.verb_prop do
    v_predict[k + k_data] = torch.zeros(5)
    l_predict[k + k_data] = torch.zeros(6)

    local v_prop, n_prop, n_idx = data.verb_prop[k]:view(-1), data.noun_prop[k], data.noun_idx[k]

    -- predicted nouns with gt verb
    local roles = verbs2roles[ info.v_gt[k + k_data] ]
    for i = 1, roles:nElement() do
      if roles[i] > 0 then
        l_predict[k + k_data][i] = n_idx[ roles[i] ]
      end
    end

    local II
    if beam_size > 1 then
      local total_prop = v_prop:clone()
      for i = 1, v_prop:nElement() do
        local roles_prop = 0
        for j = 1, role_length[i] do
          roles_prop = roles_prop + n_prop[ verbs2roles[i][j] ]
        end
        total_prop[i] = total_prop[i] + roles_prop / role_length[i]
      end

      _, II = torch.sort(total_prop, true)
    else
      _, II = torch.sort(v_prop, true)
    end
    
    for i = 1, 5 do
      v_predict[k + k_data][i] = II[i]
    end
  end

  return #data.verb_prop
end

function load_result_updatev( filename )
  local data = torch.load(filename)
  -- local top1_verb_match = torch.zeros(504)
  -- local top10_verb_match = torch.zeros(504)
  -- local top1_verb_correct = torch.zeros(504)
  -- local top10_verb_correct = torch.zeros(504)
  -- local average_idx = torch.zeros(504, 504)
  -- local average_prop = torch.zeros(504, 10)
  for k = 1, #data.verb_prop do
    v_predict[k + k_data] = torch.Tensor(5)

    local v_prop, n_prop = data.verb_prop[k], data.noun_prop[k]
    local top_prop, top_idx = data.verb_top_prop[k], data.verb_top_idx[k]
    local PP, II = torch.sort(v_prop:view(-1), true)

    -- local gt_v = math.ceil((k + k_data)/50)
    -- average_prop:add(top_prop)
    -- for i = 1, top_idx:size(1) do
    --   for j = 1, top_idx:size(2) do
    --     average_idx[i][ top_idx[i][j] ] = 1
    --   end

    --   if top_idx[i][1] == i then top1_verb_match[i] = top1_verb_match[i] + 1 end
    --   if torch.any( torch.eq(top_idx[i], i) ) then
    --     top10_verb_match[i] = top10_verb_match[i] + 1
    --   end

    --   if top_idx[i][1] == gt_v then top1_verb_correct[i] = top1_verb_correct[i] + 1 end
    --   if torch.any( torch.eq(top_idx[i], gt_v) ) then
    --     top10_verb_correct[i] = top10_verb_correct[i] + 1
    --   end
    -- end

    -- print('start beam search')
    local candidates = {}
    if beam_size > 1 then
      local graph_flag = torch.ByteTensor(#graph2verb):fill(0)
      for i = 1, #graph2verb do
        if torch.sum(graph_flag) > beam_size and #candidates >= 5 then break end
        if graph_flag[ verb2graph[ II[i] ] ] == 0 then
          graph_flag[ verb2graph[ II[i] ] ] = 1

          local tmp_prop = 0
          for r = 1, role_length[ II[i] ] do
            tmp_prop = tmp_prop + n_prop[ II[i] ][r]
          end
          tmp_prop = tmp_prop / role_length[ II[i] ]

          -- print(tmp_prop)
          for j = 1, math.min(beam_size, 10) do
            if sameGraph(II[i], top_idx[ II[i] ][j]) then
              -- print(top_prop[ II[i] ][j] .. ' ' .. tmp_prop .. ' ' .. top_idx[ II[i] ][j])
              table.insert(candidates, {v=top_idx[ II[i] ][j],
                          p=PP[i] + top_prop[ II[i] ][j] + tmp_prop})
            end
          end
        end
      end
      table.sort( candidates, compare )
    end

    -- print('#candidates ' .. #candidates)
    for v = 1, 5 do
      if beam_size > 1 then
        v_predict[k + k_data][v] = candidates[v] and candidates[v].v or candidates[1].v
      else
        v_predict[k + k_data][v] = II[v]
      end
    end
  end

  return #data.verb_prop, {top1_verb_match=top1_verb_match, top10_verb_match=top10_verb_match,
                          top1_verb_correct=top1_verb_correct, top10_verb_correct=top10_verb_correct,
                          average_prop=average_prop, average_idx=average_idx}
end

  -- local t1_verb_match = torch.zeros(504)
  -- local t10_verb_match = torch.zeros(504)
  -- local t1_verb_correct = torch.zeros(504)
  -- local t10_verb_correct = torch.zeros(504)
  -- local ave_idx = torch.zeros(504, 504)
  -- local ave_prop = torch.zeros(504, 10)
  -- local DD
for ifile = 1, num_files do

  local k_inc
  local dataname = string.format('%s_%04d.t7', save_t7_prefix, ifile)
  if opt.model == 'e4' then
    k_inc = load_result_roleonly( dataname )
  elseif opt.model == 'e5' then
    k_inc, DD = load_result_updatev( dataname )
    -- t1_verb_match:add(DD.top1_verb_match)
    -- t10_verb_match:add(DD.top10_verb_match)
    -- t1_verb_correct:add(DD.top1_verb_correct)
    -- t10_verb_correct:add(DD.top10_verb_correct)
    -- ave_idx:add(DD.average_idx)
    -- ave_prop:add(DD.average_prop)
    -- print(color.blue(k_data))
  elseif opt.model == 'e6' then

  elseif opt.model == 'e7' then

  elseif opt.model == 'e8' then
    k_inc = load_result_prgraph( dataname )
  else
    assert(false, 'Error: model is not defined!')
  end
  k_data = k_data + k_inc

end
  -- ave_prop:div(k_data)
  -- ave_prop = torch.mean(ave_prop, 1)
  -- print('t1_verb_match')
  -- print(t1_verb_match:div(k_data))
  -- print('t10_verb_match')
  -- print(t10_verb_match:div(k_data))
  -- print('t1_verb_correct')
  -- print(t1_verb_correct:div(k_data))
  -- print('t10_verb_correct')
  -- print(t10_verb_correct:div(k_data))
  -- print('ave_prop')
  -- print(ave_prop)
  -- local PP, II = torch.sort(ave_idx, 2, true)
  -- print(II[{{},{1}}])
  -- print(PP[{{},{1}}])


-- local prob = torch.zeros(10, 1):float()
-- -- local num = torch.zeros(#graph2verb, 1):float()
-- for ifile = 1, num_files do
--   local data = torch.load(string.format('%s_%04d.t7', save_t7_prefix, ifile))
 
--   for k = 1, #data.score_prop do
--     v_predict[k + k_data] = torch.Tensor(5)

--     PP, II = torch.sort(data.score_prop[k], 1, true)
--     _, JJ = torch.sort(data.verb_prop[k], 2, true)

--     if k + k_data <= num_data then
--       prob:add( PP[{{1,10},{}}] )

--       local gtg = verb2graph[ info.v_gt[k + k_data] ]

--       graph_acc[gtg][ II[1] ] = graph_acc[gtg][ II[1] ] + 1

--       graph_num[gtg] = graph_num[gtg] + 1
--     end

--     for m = 1, 5 do
--       v_predict[k + k_data][m] = JJ[ II[m][1] ][1]
--     end
--   end

--   k_data = k_data + #data.score_prop
-- end

-- print(prob:div(num_data))
-- for i = 1, #graph2verb do
--   local pre_wrong, j_wrong = 0, 0
--   for j = 1, #graph2verb do
--     if graph_acc[i][j] > pre_wrong then
--       pre_wrong, j_wrong = graph_acc[i][j], j
--     end
--   end

--   local pre_confuse, j_confuse = 0, 0
--   local total_prediction = 0
--   for j = 1, #graph2verb do
--     total_prediction = total_prediction + graph_acc[j][i]
--     if graph_acc[j][i] > pre_confuse then
--       pre_confuse, j_confuse = graph_acc[j][i], j
--     end
--   end
--   -- graph %d has %d verbs, %d sample, %d correct predict, %f acc
--   -- %d wrong prediction, index %d
--   -- %d confuse prediction, index %d
--   -- %d total_prediction
--   -- print(string.format('%d\t%d\t%d\t%d\t%.2f\t%d\t%d\t%d\t%d\t%d', 
--   --       i, #graph2verb[i], graph_num[i], graph_acc[i][i], graph_acc[i][i]/graph_num[i]*100,
--   --       j_wrong, pre_wrong, j_confuse, pre_confuse, total_prediction))
-- end
-- local num_correct = 0
-- for i = 1, #graph2verb do
--   num_correct = num_correct + graph_acc[i][i]
-- end
-- print(color.red(num_correct / num_data * 100))

info.v_predict = v_predict
if opt.model == 'e8' then
  info.predict = l_predict
end
info.logp = sum_logp
print('Calculating accuracy ...')
local Acc = utils.calc_accuracy(info)
print(string.format('VerbLoss: %.4f', info.verb_loss))
print(string.format('NounLoss: %.4f', info.noun_loss))
print(string.format('Agents top1: %.4f%%', Acc.agents_top1 * 100))
print(string.format('Places top1: %.4f%%', Acc.places_top1 * 100))
print(string.format('Value: %.4f%%', Acc.value * 100))
print(string.format('Value-all: %.4f%%', Acc.value_any * 100))
print(string.format('*verbs* top1: %.4f%%', Acc.verbs_top1 * 100))
print(string.format('Value top1: %.4f%%', Acc.value_top1 * 100))
print(string.format('Value-all top1: %.4f%%', Acc.value_any_top1 * 100))
print(string.format('*verbs* top5: %.4f%%', Acc.verbs_top5 * 100))
print(string.format('Value top5: %.4f%%', Acc.value_top5 * 100))
print(string.format('Value-all top5: %.4f%%', Acc.value_any_top5 * 100))


local hdf5 = require 'hdf5'
local confusion_matrix = torch.zeros(504, 504):float()
for i = 1, num_data do
  local v_gt_i, v_predict_i = info.v_gt[i], info.v_predict[i]
  confusion_matrix[v_gt_i][v_predict_i[1]] = confusion_matrix[v_gt_i][v_predict_i[1]] + 1
end
for i = 1, 504 do
  if i % 4 == 0 then
    local tmp = confusion_matrix[i]
    local _, II = torch.sort(tmp, true)
    confusion_matrix[i][i] = confusion_matrix[i][i] - 1
    if i == II[1] then
      confusion_matrix[i][II[2]] = confusion_matrix[i][II[2]] + 1
    else
      confusion_matrix[i][II[1]] = confusion_matrix[i][II[1]] + 1
    end
  end
end
confusion_matrix:div(50)
print(torch.sum(torch.diag(confusion_matrix))/504)

local fp = hdf5.open('./confusion_matrix.h5', 'w')
fp:write('/confusion_matrix', confusion_matrix)
fp:close()