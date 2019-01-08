require 'torch'
require 'nn'
require 'nngraph'

require 'model.RNNCriterion'
require 'model.RNNCriterionWithScore'
require 'misc.DataLoader'
require 'misc.optim_updates'
local utils = require 'misc.utils'
local model_utils = require 'misc.model_utils'
local GatedGraph = require 'model.GatedGraph'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Test a fully-connected graph model')
cmd:text()
cmd:text('Options')

-- General settings
cmd:option('-input_h5', '')
cmd:option('-split_test', '', 'dev|test')
cmd:option('-gpu_id', 0, 'which gpu to use, -1 = CPU')
cmd:option('-num_threads', 1, 'how many of threads are used to load data')
cmd:option('-save_t7_prefix', '')

-- Optimization
cmd:option('-batch_size', 256)
cmd:option('-num_updates', 0)

-- Checkpoint
cmd:option('-checkpoint_path', '/path/to/model')
cmd:option('-checkpoint_name', '')
cmd:option('-beam_size', 5)

cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU
if opt.gpu_id >= 0 then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu_id + 1) -- lua starts from 1
end

-------------------------------------------------------------------------------
-- Load the model checkpoint to evaluate
-------------------------------------------------------------------------------
assert(string.len(opt.checkpoint_name) > 0, 'must provide a model')
local checkpoint = torch.load(opt.checkpoint_path .. '/' .. opt.checkpoint_name)
-- override and collect parameters
if string.len(opt.input_h5) == 0 then opt.input_h5 = checkpoint.opt.input_h5 end
if string.len(opt.split_test) == 0 then opt.split_test = checkpoint.opt.split_test end
if opt.batch_size == 0 then opt.batch_size = checkpoint.opt.batch_size end
local fetch = {'vocab_size', 'rnn_size'}
for k,v in pairs(fetch) do 
  opt[v] = checkpoint.opt[v] -- copy over options from model
end
print(string.format('[Test checkpoint: %s]', opt.checkpoint_name))
print(string.format('rnn_size: %d, num_updates(train): %d, num_updates(test): %d', 
  opt.rnn_size, checkpoint.opt.num_updates, opt.num_updates))

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader = DataLoader{h5_file = opt.input_h5}

-------------------------------------------------------------------------------
-- Initialize the networks
-------------------------------------------------------------------------------
opt.seq_length = loader:getSeqLength()
local image_size = loader:getImageSize()  -- image feature

local protos = checkpoint.protos

-- ship the model to the GPU if desired
if opt.gpu_id >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name, proto in pairs(protos) do
  if name == 'graph' then
    print('cloning ' .. name .. ' ' .. opt.num_updates .. ' times')
    clones[name] = model_utils.clone_many_times(proto, opt.num_updates)
  else
    print('cloning ' .. name .. ' ' .. opt.seq_length .. ' times')
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length)
  end
end

-- we are ready for training/evaluating, clean up the house :)
collectgarbage()

local struct = loader:getFcStructure_roleonly()

local function forward_graph( D )
  -- dump input data
  local img_agents_k,   img_places_k,   img_verbs_k,   roles,   mask,   verb_k,   label_k = 
      D.img_agents_k, D.img_places_k, D.img_verbs_k, D.roles, D.mask, D.verb_k, D.label_k

  local num_roles   = roles:size(1)
  local num_verbs   = roles:size(2)
  local image_size  = img_verbs_k:size(2)
  local img_agents  = img_agents_k:expand(num_verbs, image_size)
  local img_places  = img_places_k:expand(num_verbs, image_size)
  local img_verbs   = img_verbs_k:expand(num_verbs, image_size)
  for i = 1, 3 do
    label_k[i] = label_k[i]:transpose(1,2):contiguous() -- swap the axes for faster indexing
    label_k[i][torch.gt(label_k[i], opt.vocab_size)] = 0 -- out of boundary
  end
  local verbs       = torch.range(1, num_verbs)
  if opt.gpu_id >= 0 then verbs = verbs:cuda() end

  local verbs_idx   = torch.zeros(num_verbs):fill(1)
  local nouns_idx   = torch.zeros(num_verbs, opt.seq_length)
  local verb_loss, noun_loss = 0, 0
  local verb_prop, noun_prop = 0, torch.zeros(num_verbs, opt.seq_length)


  local verb_state = clones.embed_verb[opt.seq_length]:forward(img_verbs)

  local rnn_state = {[0] = {}}
  for t = 1, opt.seq_length do
    if t == 1 then
      rnn_state[0][t] = clones.embed_role[t]:forward({img_places, verbs, roles[t]})
    else
      rnn_state[0][t] = clones.embed_role[t]:forward({img_agents, verbs, roles[t]})
    end
  end

  for n = 1, opt.num_updates do
    for t = 1, opt.seq_length do table.insert(rnn_state[n-1], struct[t]) end
    table.insert(rnn_state[n-1], mask)
    rnn_state[n] = clones.graph[n]:forward( rnn_state[n-1] )
  end

  local verb_predict = clones.output_verb[opt.seq_length]:forward(verb_state)
  verb_loss = clones.criterion_verb[opt.seq_length]:forward(verb_predict[{{verb_k[1]}}], verb_k)
  verb_prop = verb_predict[{{verb_k[1]}}]:float()

  local predictions
  for t = 1, opt.seq_length do
    predictions = clones.output_role[t]:forward(rnn_state[opt.num_updates][t])
    local tmp_loss = 0
    for i = 1, 3 do
      tmp_loss = tmp_loss + clones.criterion_role[t]:forward(predictions[{{verb_k[1]}}], label_k[i][t])
    end
    noun_loss = noun_loss + tmp_loss / 3
    local PP, II = torch.max(predictions, 2)
    for v = 1, num_verbs do
      nouns_idx[v][t] = II[v][1]
      noun_prop[v][t] = PP[v][1]
    end
  end

  return {verbs_idx=verbs_idx, nouns_idx=nouns_idx, vprop=verb_prop, nprop=noun_prop,
          vloss=verb_loss, nloss=noun_loss/opt.seq_length}
end

-------------------------------------------------------------------------------
-- Validation Evaluation (over an entire split)
-------------------------------------------------------------------------------
local function sample_split( split ) -- split: dev|test
  assert(split == 'dev' or split == 'test')

  -- set current phase to evaluate
  for t = 1, opt.seq_length do
    clones.embed_verb[t]:evaluate()
    clones.embed_role[t]:evaluate()
    clones.output_verb[t]:evaluate()
    clones.output_role[t]:evaluate()
  end
  for n = 1, opt.num_updates do
    clones.graph[n]:evaluate()
  end

  -- reset iterator
  loader:resetIterator(split)

  -- used for beam search
  local verbs2roles = loader:getVerbs2Roles()

  for t = 1, #struct do
    struct[t] = struct[t]:expand(verbs2roles:size(1), struct[t]:size(2))
    if opt.gpu_id >= 0 then struct[t] = struct[t]:cuda() end
  end

  local mask = torch.zeros(verbs2roles:size(1), opt.seq_length)
  mask = verbs2roles:clone()
  mask[torch.gt(mask, 1)] = 1
  if opt.gpu_id >= 0 then mask = mask:cuda() end

  verbs2roles = verbs2roles:transpose(1,2):contiguous() -- for faster indexing
  verbs2roles[torch.eq(verbs2roles, 0)] = loader:getNumRoles() + 1
  if opt.gpu_id >= 0 then verbs2roles = verbs2roles:cuda() end
  
  -- save the gt labels & predictions
  local l_length = loader:getLabelLength(split)
  local l_gt = {}
  local l_predict = {}
  local v_gt = {}
  local v_predict = {}
  local verb_loss, noun_loss = 0, 0

  local k_data = 0
  while true do
    -- get batch data
    local D = loader:getBatch{batch_size=opt.batch_size, split=split}
    local img_agents,   img_places,   img_verbs,   labels,   roles,   wrapped,   verbs = 
        D.img_agents, D.img_places, D.img_verbs, D.labels, D.roles, D.wrapped, D.verbs

    -- save gt labels & init predictions
    for k = 1, opt.batch_size do
      l_gt[k + k_data] = torch.Tensor(3, opt.seq_length)
      for j = 1, 3 do
        l_gt[k + k_data][j] = labels[j][k]
      end
      l_predict[k + k_data] = torch.Tensor(opt.seq_length)
      v_gt[k + k_data]      = verbs[k]
      v_predict[k + k_data] = torch.Tensor(5)
    end

    -- gpu data
    if opt.gpu_id >= 0 then
      img_agents  = img_agents:cuda() 
      img_places  = img_places:cuda() 
      img_verbs   = img_verbs:cuda() 
      roles = roles:cuda()
      verbs = verbs:cuda()
      for i = 1, 3 do labels[i] = labels[i]:cuda() end
    end


    local verb_prop = {}
    local noun_prop = {}
    for k = 1, opt.batch_size do

      local ret = forward_graph({
        img_agents_k=img_agents[{ {k,k} }],
        img_places_k=img_places[{ {k,k} }],
        img_verbs_k=img_verbs[{ {k,k} }],
        verb_k=verbs[{ {k,k} }],
        label_k={labels[1][{ {k,k} }], labels[2][{ {k,k} }], labels[3][{ {k,k} }]},
        roles=verbs2roles,
        mask=mask
      })

      v_predict[k + k_data] = ret.verbs_idx[{{1,5}}]:clone()
      l_predict[k + k_data] = ret.nouns_idx[{{ verbs[k] }}]:view(-1):clone()

      verb_prop[k] = ret.vprop
      noun_prop[k] = ret.nprop

      verb_loss = verb_loss + ret.vloss
      noun_loss = noun_loss + ret.nloss
    end

    k_data = k_data + opt.batch_size

    torch.save(string.format('%s_%04d.t7', opt.save_t7_prefix, k_data / opt.batch_size),
                {noun_prop=noun_prop, verb_prop=verb_prop})

    -- we have iterated over the whole batches in this split
    if wrapped then break end
  end

  return {gt=l_gt, predict=l_predict, length=l_length,
          v_gt=v_gt, v_predict=v_predict,
          verb_loss=verb_loss/k_data,
          noun_loss=noun_loss/k_data}
end


-------------------------------------------------------------------------------
-- Main
-------------------------------------------------------------------------------
print('Start testing model: ' .. opt.checkpoint_name .. ', \'' .. opt.split_test .. '\' split ')
local res = sample_split( opt.split_test )
torch.save(opt.save_t7_prefix .. '_info.t7', res)
-- print('Calculating accuracy ...')
-- local Acc = utils.calc_accuracy(res)
-- print(string.format('Agents top1: %.2f%%', Acc.agents_top1 * 100))
-- print(string.format('Places top1: %.2f%%', Acc.places_top1 * 100))
-- print(string.format('Value: %.4f%%', Acc.value * 100))
-- print(string.format('Value-all: %.4f%%', Acc.value_any * 100))
-- print(string.format('*verbs* top1: %.2f%%', Acc.verbs_top1 * 100))
-- print(string.format('Value top1: %.4f%%', Acc.value_top1 * 100))
-- print(string.format('Value-all top1: %.4f%%', Acc.value_any_top1 * 100))
-- print(string.format('*verbs* top5: %.2f%%', Acc.verbs_top5 * 100))
-- print(string.format('Value top5: %.4f%%', Acc.value_top5 * 100))
-- print(string.format('Value-all top5: %.4f%%', Acc.value_any_top5 * 100))
-- print(string.format('EvalVerbLoss: %.4f', res.verb_loss))
-- print(string.format('EvalNounLoss: %.4f', res.noun_loss))
