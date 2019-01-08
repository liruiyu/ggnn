require 'torch'
require 'nn'
require 'nngraph'

require 'model.RNNCriterion'
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
cmd:text('Train a fully-connected graph model')
cmd:text()
cmd:text('Options')

-- General settings
cmd:option('-input_h5', '/path/to/data')
cmd:option('-split_train', 'train', 'train')
cmd:option('-split_test', 'test', 'dev|test')
cmd:option('-gpu_id', 0, 'which gpu to use, -1 = CPU')
cmd:option('-num_threads', 1, 'how many of threads are used to load data')

-- Model parameters
cmd:option('-embedding_size', 1024)
cmd:option('-rnn_size', 1024)
cmd:option('-num_updates', 1, 'how many updates used in graph')
cmd:option('-vocab_size', 2000)

-- Optimization
cmd:option('-num_neg_samples', 0, 'number of negative samples')
cmd:option('-score_mul', 0)
cmd:option('-max_iters', -1)
cmd:option('-batch_size', 256)
cmd:option('-grad_clip', 1)
cmd:option('-drop_prob', 0.5)
cmd:option('-learning_rate_verb', 1e-3)
cmd:option('-learning_rate_noun', 1e-3)
cmd:option('-learning_rate_decay',0.85)
cmd:option('-learning_rate_decay_after', 1, 'epoch')
cmd:option('-optim', 'rmsprop', 'rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-optim_alpha', 0.9, 'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta', 0.999, 'beta used for adam')
cmd:option('-optim_epsilon', 1e-8, 'avoid numerical problem')

-- Checkpoint
cmd:option('-save_checkpoint_every', 2000)
cmd:option('-checkpoint_path', '/path/to/model')
cmd:option('-losses_log_every', 100)

cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
print('[Train a fully-connected graph model.]')
print('embedding_size       ' .. opt.embedding_size)
print('rnn_size             ' .. opt.rnn_size)
print('drop_prob            ' .. opt.drop_prob)
print('num_updates          ' .. opt.num_updates)
print('num_neg_samples      ' .. opt.num_neg_samples)
print('learning_rate_verb   ' .. opt.learning_rate_verb)
print('learning_rate_noun   ' .. opt.learning_rate_noun)
-- print('score_mul            ' .. opt.score_mul)
print('[-------------------------------]')
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU
if opt.gpu_id >= 0 then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu_id + 1) -- lua starts from 1
end

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader = DataLoader{h5_file = opt.input_h5}

-------------------------------------------------------------------------------
-- Initialize the networks
-------------------------------------------------------------------------------
local do_random_init  = true
opt.seq_length        = loader:getSeqLength()
local image_size      = loader:getImageSize()  -- image feature
local num_verbs       = loader:getNumVerbs() -- max verb index
local num_roles       = loader:getNumRoles() -- max role index
local vocab_size      = opt.vocab_size     -- word distribution
local embedding_size  = opt.embedding_size -- image & role embedding size
local rnn_size        = opt.rnn_size -- the size of hidden state in graph

local protos = {}
protos.embed_verb   = nn.Sequential()--:add(nn.BatchNormalization(image_size))
                              :add(nn.Linear(image_size, embedding_size))
                              :add(nn.ReLU())
local img_embed     = nn.Sequential()--:add(nn.BatchNormalization(image_size))
                              :add(nn.Linear(image_size, embedding_size))
local ltable_verbs  = nn.Sequential():add(nn.LookupTable(num_verbs, embedding_size))
local ltable_roles  = nn.Sequential():add(nn.LookupTable(num_roles + 1,  embedding_size))
local parallel      = nn.ParallelTable():add(img_embed):add(ltable_verbs):add(ltable_roles)
protos.embed_role   = nn.Sequential():add(parallel):add(nn.CMulTable())
                              :add(nn.ReLU())
                              --:add(nn.Linear(embedding_size, rnn_size)):add(nn.ReLU())
print(protos.embed_verb)
print(protos.embed_role)
protos.graph        = GatedGraph.graph(rnn_size, 0, opt.seq_length)
protos.output_verb  = nn.Sequential():add(nn.Dropout(opt.drop_prob))
                              :add(nn.Linear(rnn_size, num_verbs)):add(nn.LogSoftMax())
protos.output_role  = nn.Sequential():add(nn.Dropout(opt.drop_prob))
                              :add(nn.Linear(rnn_size, vocab_size)):add(nn.LogSoftMax())
print(protos.output_verb)
print(protos.output_role)
protos.criterion_verb   = nn.RNNCriterion() -- each image must have a verb
protos.criterion_role   = nn.RNNCriterion() -- nouns may not exist

-- ship the model to the GPU if desired
if opt.gpu_id >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

-- put the above things into one flattened parameters tensor
params_verb, grad_params_verb = model_utils.combine_all_parameters(
                                  protos.embed_verb, protos.output_verb)
params_noun, grad_params_noun = model_utils.combine_all_parameters(
                                  protos.embed_role, protos.graph, protos.output_role)

-- initialization
if do_random_init then
  params_verb:uniform(-0.08, 0.08) -- small uniform numbers
  params_noun:uniform(-0.08, 0.08) -- small uniform numbers
end
ltable_roles.modules[1].weight[ltable_roles.modules[1].weight:size(1)]:zero()

print('number of parameters in the model: ' .. params_verb:nElement()+params_noun:nElement())
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
for t = 1, #struct do
  struct[t] = struct[t]:expand(opt.batch_size, struct[t]:size(2))
  if opt.gpu_id >= 0 then struct[t] = struct[t]:cuda() end
end

-------------------------------------------------------------------------------
-- Loss function (only for a minibatch)
-------------------------------------------------------------------------------
local function lossFun( split ) -- split: train
  -- set current phase to training
  for t = 1, opt.seq_length do
    clones.embed_verb[t]:training()
    clones.embed_role[t]:training()
    clones.output_verb[t]:training()
    clones.output_role[t]:training()
  end
  for n = 1, opt.num_updates do
    clones.graph[n]:training()
  end

  -- zero out the gradient
  grad_params_verb:zero()
  grad_params_noun:zero()

  -- get minibatch
  local D = loader:getBatch{batch_size=opt.batch_size, split=split, num_neg_samples=opt.num_neg_samples}
  local img_agents,   img_places,   img_verbs,   labels,   roles,   wrapped,   verbs,   scores= 
      D.img_agents, D.img_places, D.img_verbs, D.labels, D.roles, D.wrapped, D.verbs, D.scores

  -- create mask
  local mask = torch.zeros(opt.batch_size, opt.seq_length)
  mask = roles:clone()
  mask[torch.gt(mask, 1)] = 1

  -- manipulate data
  roles = roles:transpose(1,2):contiguous() -- for faster indexing
  roles[torch.eq(roles, 0)] = num_roles + 1 -- we will set the gradient to zero
  labels = labels:transpose(1,2):contiguous() -- swap the axes for faster indexing
  labels[torch.gt(labels, opt.vocab_size)] = 0 -- out of boundary
  if opt.gpu_id >= 0 then
    img_agents  = img_agents:cuda() 
    img_places  = img_places:cuda() 
    img_verbs   = img_verbs:cuda() 
    labels = labels:cuda()
    roles  = roles:cuda()
    verbs  = verbs:cuda()
    mask   = mask:cuda()
    scores = scores:cuda()
  end

  ---------------------------------------------------------------------------
  -- Forward pass
  ---------------------------------------------------------------------------
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

  local verb_loss, role_loss = 0, 0

  local verb_predict = clones.output_verb[opt.seq_length]:forward(verb_state)
  verb_loss = clones.criterion_verb[opt.seq_length]:forward(verb_predict, verbs)

  local predictions = {} -- softmax outputs
  for t = 1, opt.seq_length do
    predictions[t] = clones.output_role[t]:forward(rnn_state[opt.num_updates][t])
    role_loss = role_loss + clones.criterion_role[t]:forward(predictions[t], labels[t])
  end
  role_loss = role_loss / opt.seq_length

  ---------------------------------------------------------------------------
  -- Backward pass
  ---------------------------------------------------------------------------
  local drnn_state = {[opt.num_updates] = {}}
  -- backprop through loss, and softmax/linear
  for t = 1, opt.seq_length do
    local dprediction_t = clones.criterion_role[t]:backward(predictions[t], labels[t])
    drnn_state[opt.num_updates][t] = clones.output_role[t]:backward(rnn_state[opt.num_updates][t], dprediction_t)
  end

  local dverb_predict = clones.criterion_verb[opt.seq_length]:backward(verb_predict, verbs)
  local dverb_state   = clones.output_verb[opt.seq_length]:backward(verb_state, dverb_predict)

  -- backprop through graph
  for n = opt.num_updates, 1, -1 do
    local lst = clones.graph[n]:backward( rnn_state[n-1], drnn_state[n] )
    drnn_state[n-1] = {}
    for t = 1, opt.seq_length do table.insert(drnn_state[n-1], lst[t]) end
  end

  for t = 1, opt.seq_length do
    if t == 1 then
      clones.embed_role[t]:backward({img_places, verbs, roles[t]}, drnn_state[0][t])
    else
      clones.embed_role[t]:backward({img_agents, verbs, roles[t]}, drnn_state[0][t])
    end
  end

  clones.embed_verb[opt.seq_length]:backward(img_verbs, dverb_state)

  -- clip gradient element-wise
  grad_params_verb:clamp(-opt.grad_clip, opt.grad_clip)
  grad_params_noun:clamp(-opt.grad_clip, opt.grad_clip)

  -- return the loss
  local losses = { verb_loss = verb_loss, 
                   role_loss = role_loss, wrapped = wrapped }
  return losses
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local loss0
local loss_history = {}
local optim_state_verb = {}
local optim_state_noun = {}
local learning_rate_verb = opt.learning_rate_verb
local learning_rate_noun = opt.learning_rate_noun
local iter = 1
local epoch = 1

print('Start training \'' .. opt.split_train .. '\' split ')
while true do

  -- eval loss/gradient
  local losses = lossFun(opt.split_train, iter)
  if iter % opt.losses_log_every == 0 then 
    loss_history[iter] = losses.role_loss
    print(string.format('iter %d: verb_loss=%.2f, role_loss=%.2f, ', 
                      iter, losses.verb_loss, losses.role_loss))
  end

  -- save checkpoint
  if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then
    -- print(string.format('eval_split %s ...', opt.split_test))
    -- local eval_losses = eval_split(opt.split_test)
    -- print(string.format('\tITER %d: loss: %.4f', iter, eval_losses.role_loss))

    local savefile = string.format('%s/rnn_iter_%d.t7', opt.checkpoint_path, iter)
    print('saving checkpoint to ' .. savefile)
    local checkpoint = {  protos = protos,
                          opt = opt,
                          iter = iter }
    torch.save(savefile, checkpoint)
  end

  -- decay learning rate
  learning_rate_verb = opt.learning_rate_verb * torch.pow(1-(iter/opt.max_iters), 0.9)
  learning_rate_noun = opt.learning_rate_noun * torch.pow(1-(iter/opt.max_iters), 0.9)
  -- if losses.wrapped then
  --   epoch = epoch + 1
  --   if (epoch > opt.learning_rate_decay_after and opt.learning_rate_decay_after >= 0) then
  --     learning_rate_verb = learning_rate_verb * opt.learning_rate_decay
  --     learning_rate_noun = learning_rate_noun * opt.learning_rate_decay
  --     print(string.format('decay learning at epoch %d, verb_lr=%.2f*1e-3, noun_lr=%.2f*1e-3',
  --                             epoch, learning_rate_verb*1e3, learning_rate_noun*1e3))
  --   end
  -- end

  -- perform a parameter update
  if opt.optim == 'rmsprop' then
    rmsprop(params_verb, grad_params_verb, learning_rate_verb, opt.optim_alpha, opt.optim_epsilon, optim_state_verb)
    rmsprop(params_noun, grad_params_noun, learning_rate_noun, opt.optim_alpha, opt.optim_epsilon, optim_state_noun)
  elseif opt.optim == 'adagrad' then
    adagrad(params_verb, grad_params_verb, learning_rate_verb, opt.optim_epsilon, optim_state_verb)
    adagrad(params_noun, grad_params_noun, learning_rate_noun, opt.optim_epsilon, optim_state_noun)
  elseif opt.optim == 'sgd' then
    sgd(params_verb, grad_params_verb, opt.learning_rate_verb)
    sgd(params_noun, grad_params_noun, opt.learning_rate_noun)
  elseif opt.optim == 'sgdm' then
    sgdm(params_verb, grad_params_verb, learning_rate_verb, opt.optim_alpha, optim_state_verb)
    sgdm(params_noun, grad_params_noun, learning_rate_noun, opt.optim_alpha, optim_state_noun)
  elseif opt.optim == 'sgdmom' then
    sgdmom(params_verb, grad_params_verb, learning_rate_verb, opt.optim_alpha, optim_state_verb)
    sgdmom(params_noun, grad_params_noun, learning_rate_noun, opt.optim_alpha, optim_state_noun)
  elseif opt.optim == 'adam' then
    adam(params_verb, grad_params_verb, learning_rate_verb, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state_verb)
    adam(params_noun, grad_params_noun, learning_rate_noun, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state_noun)
  else
    error('bad option opt.optim')
  end

  -- stopping criterions
  iter = iter + 1
  if iter % 10 == 0 then collectgarbage() end -- do this once in a while
  if loss0 == nil then loss0 = losses.role_loss end
  if losses.role_loss > loss0 * 20 then
    print('loss seems to be exploding, quitting.')
    break
  end
  if opt.max_iters > 0 and iter > opt.max_iters then break end -- stopping criterion

end

