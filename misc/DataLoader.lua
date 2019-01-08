require 'hdf5'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
	-- open the hdf5 file
	print('DataLoader: loading hdf5 file from: \n\t' .. opt.h5_file)
	self.h5_file = hdf5.open(opt.h5_file, 'r')

	-- extract image feature size
  if not opt.wo_preload then
    local tmp_data  = self.h5_file:read('/images'):all()
    tmp_data        = self.h5_file:read('/imgs_places'):all()
    tmp_data        = self.h5_file:read('/imgs_verbs'):all() -- speed up hdf5
  end
	local data_size = self.h5_file:read('/imgs_verbs'):dataspaceSize()
	self.num_images = data_size[1]
	self.image_size = data_size[2]
	print(string.format('DataLoader: read %d images of size %d', self.num_images, self.image_size))

  -- load the information about the images to RAM (should be small enough)
  self.image_info = self.h5_file:read('/image_info'):all():squeeze()
  -- load labels
  self.labels = self.h5_file:read('/labels'):all()
  self.seq_length = self.labels:size(2)
  print(string.format('DataLoader: read %d labels of size %d', self.labels:size(1), self.seq_length))
  self.label_start_ix = self.h5_file:read('/label_start_ix'):all():squeeze()
  self.label_end_ix = self.h5_file:read('/label_end_ix'):all():squeeze()
  self.label_length = self.h5_file:read('/label_length'):all():squeeze()
  self.verbs = self.h5_file:read('/verbs'):all():squeeze() -- size: num_images and \in [1, 504]
  self.num_verbs = torch.max(self.verbs)
  self.roles = self.h5_file:read('/roles'):all() -- size: num_images x 6 and \in [1, 190]
  self.num_roles = torch.max(self.roles)
  self.verbs2roles = self.h5_file:read('/verbs2roles'):all() -- size: 504 x 6
  self.graph2verb, self.verb2graph, self.graphs = build_graphs( self.verbs2roles )
  self.role_connections = pairwise_roles( self.verbs2roles )

  -- separate out indexes for each of the provided split
	self.split_idx = {['train']={},['dev']={},['test']={}}
	self.iterators = {['train']=1,['dev']=1,['test']=1}
	for i = 1, self.image_info:nElement() do
    local info = self.image_info[i]
		if info == 1 then
  		table.insert(self.split_idx['train'], i)
		elseif info == 2 then
  		table.insert(self.split_idx['dev'], i)
		elseif info == 3 then
  		table.insert(self.split_idx['test'], i)
    else
      -- this is for small train/dev/test data
		end
	end
--	for k, v in pairs(self.split_idx) do
--    print(string.format('DataLoader: assigned %d images to split %s', #v, k))
--	end
  self.fetch = { 	-- shuffle the training data
  					     ['train']		= torch.randperm(#self.split_idx['train']), 
  					     -- do nothing with the validation data
  					     ['dev']			= torch.range(1, #self.split_idx['dev']),
  					     ['test']		  = torch.range(1, #self.split_idx['test']) }
end

function build_graphs( verbs2roles )
  local v2r = torch.sort(verbs2roles, true)
  local graphs_tmp  = torch.rand(v2r:size())
  local verb2graph  = torch.rand(v2r:size(1))
  local graph2verb  = {}
  local g_flags = {}
  local k = 0
  for i = 1, v2r:size(1) do
      local tmp_str = "g"
      for j = 1, v2r:size(2) do
          tmp_str = string.format('%s_%03d', tmp_str, v2r[i][j])
      end
      if g_flags[tmp_str] == nil then
          k = k + 1
          g_flags[tmp_str] = k;
          graph2verb[k] = {}
          graphs_tmp[k] = verbs2roles[i]:clone()
      end
      verb2graph[i] = g_flags[tmp_str]
      table.insert(graph2verb[ g_flags[tmp_str] ], i)
  end
  local graphs = graphs_tmp[{{1,k},{}}]:clone()

  return graph2verb, verb2graph, graphs
end

function pairwise_roles( verbs2roles )
  local num_roles = torch.max(verbs2roles)
  local role_connections = torch.zeros(num_roles, num_roles)
  for k = 1, verbs2roles:size(1) do
    local v2r_k = verbs2roles[k]
    for i = 1, verbs2roles:size(2) do
      for j = i + 1, verbs2roles:size(2) do
        if v2r_k[i] ~= 0 and v2r_k[j] ~= 0 and v2r_k[i] ~= v2r_k[j] then
          role_connections[v2r_k[i]][v2r_k[j]] = 1
          role_connections[v2r_k[j]][v2r_k[i]] = 1
        end
      end
    end
  end
  return role_connections
end

function DataLoader:sample_negative(split, v)
  local split_idx = self.split_idx[split]
  local max_index = #split_idx
  local is_found = false
  local ver, rol, lab
  while is_found == false do
    local ix = split_idx[ torch.random(1, max_index) ]
    ver = self.verbs[ix]
    if self.verb2graph[v] == self.verb2graph[ver] then
      -- continue
    else
      is_found = true
      local ix1 = self.label_start_ix[ix]
      local ix2 = self.label_end_ix[ix]
      local ncap = ix2 - ix1 + 1 -- number of labels available for this image
      local seq = self.labels[{{ix1,ix2}, {1,self.seq_length}}]
      lab = seq[{{ torch.random(1,ncap) },{}}]:clone()
      rol = self.roles[ix]:clone()
    end
  end
  return lab, ver, rol
end

function DataLoader:getBatch(opt)
	local batch_size = opt.batch_size
	local split = opt.split
  local num_neg_samples = opt.num_neg_samples
	local label_per_image = 3
	local split_idx = self.split_idx[split]
	local fetch = self.fetch[split]

  local img_agents  = torch.Tensor(batch_size, self.image_size)
  local img_places  = torch.Tensor(batch_size, self.image_size)
  local img_verbs   = torch.Tensor(batch_size, self.image_size)
  local roles       = torch.Tensor(batch_size, self.seq_length)
  local verbs       = torch.Tensor(batch_size)
  local scores      = torch.Tensor(batch_size)
	local labels
	if split == 'train' then
		labels = torch.Tensor(batch_size, self.seq_length)
	else
		labels = {}
		for i = 1, label_per_image do 
			table.insert(labels, torch.Tensor(batch_size, self.seq_length))
		end
	end
	local max_index = #split_idx
	local wrapped = false

	local b = 1
	while b <= batch_size do

		local ri = self.iterators[split] -- get next index from iterator
		local ix = split_idx[ fetch[ri] ]
    assert(ix ~= nil, 'bug: split ' .. split .. ' was accessed out of bounds with ' .. ri)
		local ri_next = ri + 1 -- increment iterator
		if ri_next > max_index then 
			ri_next, wrapped = 1, true -- wrap back around
			if split == 'train' then -- reshuffle data
				self.fetch[split] = torch.randperm(max_index)
				fetch = self.fetch[split]
        print('DataLoader: reshuffle the split \'' .. split .. '\' data')
			end
		end
		self.iterators[split] = ri_next

		-- fetch image feature
    local img_a = self.h5_file:read('/images'):partial({ix,ix},{1,self.image_size})
    local img_p = self.h5_file:read('/imgs_places'):partial({ix,ix},{1,self.image_size})
    local img_v = self.h5_file:read('/imgs_verbs'):partial({ix,ix},{1,self.image_size})
    -- get verbs and roles
    local ver, rol = self.verbs[ix], self.roles[ix]

    -- fetch sequential labels
    local ix1 = self.label_start_ix[ix]
    local ix2 = self.label_end_ix[ix]
    local ncap = ix2 - ix1 + 1 -- number of labels available for this image
    assert(ncap > 0, 'an image does not have any label')
    assert(ncap == label_per_image, 'an image does not have enough labels')
    local seq = self.labels[{{ix1,ix2}, {1,self.seq_length}}]:clone()
    if split == 'train' then
    	for i = 1, num_neg_samples + 1 do
    		if b > batch_size then break end
        if i == 1 then
          labels[b] = seq[{{ torch.random(1,ncap) },{}}]
          verbs[b]  = ver
          roles[b]  = rol:clone()
          img_agents[b] = img_a:clone()
          img_places[b] = img_p:clone()
          img_verbs[b]  = img_v:clone()
          scores[b] = 1 -- positive
        else
          scores[b] = 0 -- negative
          if torch.random(0, 1) == 0 then -- same image, diff structure
            labels[b], verbs[b], roles[b] = self:sample_negative(split, ver)
            img_agents[b] = img_a:clone()
            img_places[b] = img_p:clone()
            img_verbs[b]  = img_v:clone()
          else -- same structure, diff image
            labels[b] = seq[{{ torch.random(1,ncap) },{}}]
            verbs[b]  = ver
            roles[b]  = rol:clone()
            local randix = ix
            while (self.verb2graph[ self.verbs[randix] ] == self.verb2graph[ver]) do
              randix = split_idx[ torch.random(1, max_index) ]
            end
            img_agents[b] = self.h5_file:read('/images'):partial({randix,randix},{1,self.image_size})
            img_places[b] = self.h5_file:read('/imgs_places'):partial({randix,randix},{1,self.image_size})
            img_verbs[b]  = self.h5_file:read('/imgs_verbs'):partial({randix,randix},{1,self.image_size})
          end
        end
    		b = b + 1
    	end
    else
    	for i = 1, label_per_image do
    		labels[i][b] = seq[{ {i},{} }]
    	end
      verbs[b]  = ver
      roles[b]  = rol:clone()
      img_agents[b] = img_a:clone()
      img_places[b] = img_p:clone()
      img_verbs[b]  = img_v:clone()
      b = b + 1
    end

	end

  return {img_agents=img_agents, img_places=img_places, img_verbs=img_verbs, 
        scores=scores, labels=labels, verbs=verbs, roles=roles, wrapped=wrapped}
end

function DataLoader:getLabelLength(split)
  -- used for validation
  assert(split == 'dev' or split == 'test')
  local split_idx = self.split_idx[split]
  local label_length = torch.Tensor( #split_idx )
  for i = 1, #split_idx do
    label_length[i] = self.label_length[ split_idx[i] ]
  end
  return label_length
end

function DataLoader:resetIterator(split)
  self.iterators[split] = 1
end

function DataLoader:getSplitSize(split)
  return #self.split_idx[split]
end

function DataLoader:getImageSize()
  return self.image_size
end

function DataLoader:getSeqLength()
  return self.seq_length
end

function DataLoader:getNumVerbs()
  return self.num_verbs
end

function DataLoader:getNumRoles()
  return self.num_roles
end

function DataLoader:getVerbs2Roles()
  return self.verbs2roles
end

function DataLoader:getGraphs()
  return self.graphs
end

function DataLoader:getConversions()
  return self.graph2verb, self.verb2graph
end

function DataLoader:getFcStructure_prgraph()
  local struct = torch.ones(self.num_roles + 1, self.num_roles + 1)
  struct[1][1] = 0 -- no self loop
  struct[{ {2,-1},{2,-1} }]:copy(self.role_connections)
  return torch.split(struct, 1)
end

function DataLoader:getFcStructure_roleonly()
  local struct = {}
  struct[1] = torch.Tensor({{0,1,1,1,1,1}}) -- incoming edge of node 1
  struct[2] = torch.Tensor({{1,0,1,1,1,1}})
  struct[3] = torch.Tensor({{1,1,0,1,1,1}})
  struct[4] = torch.Tensor({{1,1,1,0,1,1}})
  struct[5] = torch.Tensor({{1,1,1,1,0,1}})
  struct[6] = torch.Tensor({{1,1,1,1,1,0}})
  return struct
end

function DataLoader:getFcStructure_tree_rebuttal()
  local struct = {}
  struct[1] = torch.Tensor({{0,0,1,1,1,1}}) -- incoming edge of node 1
  struct[2] = torch.Tensor({{0,0,1,1,1,1}})
  struct[3] = torch.Tensor({{1,1,0,0,0,0}})
  struct[4] = torch.Tensor({{1,1,0,0,0,0}})
  struct[5] = torch.Tensor({{1,1,0,0,0,0}})
  struct[6] = torch.Tensor({{1,1,0,0,0,0}})
  return struct
end

function DataLoader:getFcStructure_chain_rebuttal()
  local struct = {}
  struct[1] = torch.Tensor({{0,1,0,0,0,0}}) -- incoming edge of node 1
  struct[2] = torch.Tensor({{1,0,1,0,0,0}})
  struct[3] = torch.Tensor({{0,1,0,1,0,0}})
  struct[4] = torch.Tensor({{0,0,1,0,1,0}})
  struct[5] = torch.Tensor({{0,0,0,1,0,1}})
  struct[6] = torch.Tensor({{0,0,0,0,1,0}})
  return struct
end

function DataLoader:getFcStructure_roleonly_tree()
  local struct = {}
  struct[1] = torch.Tensor({{0,1,0,0,0,0}}) -- incoming edge of node 1
  struct[2] = torch.Tensor({{1,0,0,0,0,0}})
  struct[3] = torch.Tensor({{1,1,0,0,0,0}})
  struct[4] = torch.Tensor({{1,1,0,0,0,0}})
  struct[5] = torch.Tensor({{1,1,0,0,0,0}})
  struct[6] = torch.Tensor({{1,1,0,0,0,0}})
  return struct
end

function DataLoader:getFcStructure_updateverb()
  local struct = {}
  struct[1] = torch.Tensor({{0,1,1,1,1,1,1}}) -- incoming edge of node 1
  struct[2] = torch.Tensor({{1,0,1,1,1,1,1}})
  struct[3] = torch.Tensor({{1,1,0,1,1,1,1}})
  struct[4] = torch.Tensor({{1,1,1,0,1,1,1}})
  struct[5] = torch.Tensor({{1,1,1,1,0,1,1}})
  struct[6] = torch.Tensor({{1,1,1,1,1,0,1}})
  struct[7] = torch.Tensor({{1,1,1,1,1,1,0}})
  return struct
end

function DataLoader:getFcStructure_updateverb_tree()
  local struct = {}
  struct[1] = torch.Tensor({{0,1,1,0,0,0,0}}) -- incoming edge of node 1
  struct[2] = torch.Tensor({{1,0,1,1,1,1,1}})
  struct[3] = torch.Tensor({{1,1,0,1,1,1,1}})
  struct[4] = torch.Tensor({{0,1,1,0,1,1,1}})
  struct[5] = torch.Tensor({{0,1,1,1,0,1,1}})
  struct[6] = torch.Tensor({{0,1,1,1,1,0,1}})
  struct[7] = torch.Tensor({{0,1,1,1,1,1,0}})
  return struct
end

function DataLoader:getFcStructure_updateverb_tree_v2()
  local struct = {}
  struct[1] = torch.Tensor({{0,1,1,0,0,0,0}}) -- incoming edge of node 1
  struct[2] = torch.Tensor({{1,0,1,1,1,1,1}})
  struct[3] = torch.Tensor({{1,1,0,1,1,1,1}})
  struct[4] = torch.Tensor({{1,1,1,0,1,1,1}})
  struct[5] = torch.Tensor({{1,1,1,1,0,1,1}})
  struct[6] = torch.Tensor({{1,1,1,1,1,0,1}})
  struct[7] = torch.Tensor({{1,1,1,1,1,1,0}})
  return struct
end

function DataLoader:getFcStructure_updateverb_tree_v3()
  local struct = {}
  struct[1] = torch.Tensor({{0,0,0,0,0,0,0}}) -- incoming edge of node 1
  struct[2] = torch.Tensor({{1,0,0,0,0,0,0}})
  struct[3] = torch.Tensor({{1,0,0,0,0,0,0}})
  struct[4] = torch.Tensor({{1,1,1,0,0,0,0}})
  struct[5] = torch.Tensor({{1,1,1,0,0,0,0}})
  struct[6] = torch.Tensor({{1,1,1,0,0,0,0}})
  struct[7] = torch.Tensor({{1,1,1,0,0,0,0}})
  return struct
end

function DataLoader:getFcStructure_updatescore()
  local struct = {}
  struct[1] = torch.Tensor({{0,1,1,1,1,1,1,1}}) -- incoming edge of node 1
  struct[2] = torch.Tensor({{0,0,1,1,1,1,1,1}})
  struct[3] = torch.Tensor({{0,1,0,1,1,1,1,1}})
  struct[4] = torch.Tensor({{0,1,1,0,1,1,1,1}})
  struct[5] = torch.Tensor({{0,1,1,1,0,1,1,1}})
  struct[6] = torch.Tensor({{0,1,1,1,1,0,1,1}})
  struct[7] = torch.Tensor({{0,1,1,1,1,1,0,1}})
  struct[8] = torch.Tensor({{0,1,1,1,1,1,1,0}})
  return struct
end