torch:
	nn:
		Sequential()
		Module:  #继承nn.Module 
			children() 
			named_children()
			modules()
			named_modules()   # 将模型分块，方便冻结每一小部分
		ModuleList()          # 对模型的部分结构进行封装。
		ModeleDict() 
			
		
		AdaptiveMaxPool2d()
		Conv2d()
		MaxPool2d()
		Dropout2d()
		Flatten()
		Linear()
		Embedding()
		
		ReLU()
		Sigmoid()
		Tanh()
		Softmax()
		BatchNorm2d()
		GroupNorm()
		
		Parameter() #nn.Parameter 具有 requires_grad = True 属性
		ParameterList()
		ParameterDict()
		
		BCELoss()
		MSELoss()
		CrossEntropyLoss()
	
		functional: # 各种功能组件的函数实现
			relu()
			sigmoid()
			tanh()
			softmax()
			
			linear()
			conv2d()
			max_pool2d()
			dropout2d()
			embedding()
			
			binary_cross_entropy()
			mse_loss()
			L1Loss()
			SmoothL1Loss()
			cross_entropy()
	
	
	optim:
		Adam()
		SGD()
	
	no_grad()
	autograd.grad(y,x,create_graph=True)
	autograd.Function
	

	save():
		save(net.state_dict(), path) # 参数
		save(net, path)	             # 参数+模型
		
		# 若为 torchkeras.Model封装后的
		save(model.net.modules.state_dict(), path)
	load()
		

	# 构造tensor
	tensor()  #(requires_grad=True)
	ones_like()
	zeros_like()
	ones()
	zeros()
	eye()
	diag()
	normal()
	randn()
	rand()
	arange()
	linspace()
	randperm()
	fill()
	Size()
	from_numpy() # tensor与array 公用内存

	
	# 切片
	index_select():
		torch.index_select(features, dim=1, index=torch.tensor([0,5,9]))
	index_fill()
	take():
		torch.take(features, torch.tensor(1, 8, 19)) #将数据看作是一维的
	masked_select()
	masked_fill()
	where()
	
	# 维度变换
	reshape()
	squeeze()
	unsqueeze()
	transpose()
	
	#合并分割
	cat()
	stack()
	split()
	
	#运算
	max()
	min()
	round()
	floor()
	ceil()
	trunc()
	remainder()
	clamp()
	
	add()
	pow()
	sum()
	mean()
	cos()
	sin()
	log()
	std()
	var()
	median()
	prod()
	
	topk()
	sort()
	inverse()
	trace()
	norm()
	det()
	eig()
	qr()
	svd()
	
	float64()
	float32()
	float
	float16()
	int64()
	int32()
	bool()
	
	IntTensor()
	BoolTensor()
	LongTensor()

	utils:
		data:
			Dataset       # 自定义数据集
			TensorDataset # 将numpy 或 DataFrame 数据转换为tensor
			random_split()
			
			DataLoader
			
			RandomSampler
			SequentialSampler
			BatchSampler

		tensorboard:
			SummaryWriter
			
	
	cuda:
		is_available()
		device_count()
		empty_cache()
	device():
		model.to(device)
		features.to(device)
	nn:
		DataParallel(model)  #包装为并行风格模型

torchvision:
	transforms:
		Compose()
		ToTener()
		RandomVerticalFlip()()
		RandomHorizontalFlip(), #随机水平翻转
		RandomRotation(45)(img)

	datasets:
		ImageFolder()  #根据图片目录创建图片数据集
	
	utils:
		make_grid()    #将多张图拼接成一张图，中间用黑色网格分割
	
	
torchkeras：
	summary(net,input_shape)
	Model:      #继承torchkeras.Model的方式构建网络, 用类风格进行训练网络
		model.compile(loss_func, optimizer, metrics_dict)  #可指定device
		model.fit()
		model.evaluate()
		model.predict()
		
	torchkeras.Model(net) # net为继承nn.Module
		model.summary(input_shape)
		
	nn.DataParallel(net)  # net为继承nn.Module
	torchkeras.Model(net)


# 实例对象的操作
tensor:
	dim()
	shape()
	size()
	view()
	t()
	reshape()
	is_contiguous()
	contiguous()
	numpy()
	cpu().numpy()
	add_()
	clone()
	item()
	tolist()
	index_select()
	
	grad
	grad.zero_()
	grad_fn
	is_leaf
	
	
net:
	add()
	train()
	forward()
	eval()
	state_dict()
	load_state_dict()
	
optimizer:
	zero_grad()
	zeros()
	step()

loss:
	backward()
	retain_grad()
	
writer:
	add_graph(net, input_to_model= torch.rand(10,2))
	add_scalar()
	add_histogram()
	add_image()
	add_images()
	add_figure()  # 是plt绘制的图表
	close()
	
	
tensorboard：
	notebook:  # tensorboard --logdir ./data/tensorboard
		list()
		start()
