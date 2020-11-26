tensorflow:
	keras:
		datasets:
		
		preprocessing:
			image:
				flow_from_directory()   # 从文件中读取图片的generator
				
			
		models:   # 都是继承tf.Module实现的
			Sequential()
				# add()
			
			load_model()  
			model_from_json()
			
			Model(inputs, outputs)  # 构建model实例对象 用inputs,outputs 
				# inputs = layers.Input(shape=[MAX_LEN])
				# x = layers.Embedding(MAX_WORDS,7)(inputs)
				# outputs = layers.Dense(1,activation = "sigmoid")(x)
				# model = models.Model(inputs = inputs,outputs = outputs)
				
			
			Model  # 自定义网络模型时，继承tf.keras.models.Model，重写Build()和Call()
			
			
		layers:    # 都是继承tf.Module实现的
			Input()
			Conv2D()
			MaxPool2D()
			Dropout()
			Flatten()
			Dense()
			Activation()  # 显式添加layers.Activation 激活层
			BatchNormalization()
			
			Layer   # 自定义模型层时，可继承tf.keras.layers.Layer 这个基类 重写Build()和Call()、get_config()
			
			
		regularizers:
			l1()
			l2()
			l1_l2()
			
		
		losses:  # 函数方式和类方式
			binary_crossentropy()
			mean_squared_error() 
			mean_absolute_error
			
			Loss     # 自定义损失函数时，对tf.keras.losses.Loss进行子类化，重写call()
			
			
		metrics:
			Mean()
			MeanAbsoluteError()
			MeanSquaredError()
			AUC()
			Rrcall()
		
			Metric   # 自定义评估指标时，对tf.keras.metrics.Metric进行子类化,重写 __init__()、update_state()、result()
			
		optimizers:
			Adam()
			SGD()
		
		callbacks:  
		#tf.keras的回调函数实际上是一个类，一般是在model.fit()时作为参数指定。⽤于控制在训练过程开始或者在训练过程结束，在每个epoch训练开始或者训练结束，在每个batch训练开始或者训练结束时执⾏
		#⼀些操作，例如收集⼀些⽇志信息，改变学习率等超参数，提前终⽌训练过程等等。

			History()
			TensorBoard()
			LearningRateScheduler()
			
			Callback  # 自定义评估指标时，对tf.keras.callbacks.Callback 进子行类化,重写 on_epoch_begin(), on_epoch_end()
		
		constraints：
			MaxNorm()
			
	
	nn:
		sigmoid()
		softmax()
		tanh()
		relu()
		leaky_relu()
		elu()
		selu()
		swish()
		gelu()
		
	
	
	# tf.data
	data:
		Dataset:
			list_files()               #从文件路径构建数据管道
			from_tensor_slices((x,y))  #注意tuple 还是 list
			from_generator()
		
		experimental:
			AUTOTUNE
			make_csv_dataset()   #从csv⽂件构建数据管道
		
		TextLineDataset()
		
	io:
		read_file()
		decode_jpeg()
		resize()
		
		TFRecordWriter()                    # 构建tfrecord
		
		FixedLenFeature([], tf.string)      # 解析tfrecord时，需要的描述量 dict
		parse_single_example()              # 解析tfrecord中的每个example
		
	
	train:                                  # 构建tfrecord， 构建Example、Features、Feature
		Example()
			#example.SerializeToString()    # 序列化
		Features()
		Feature()
		Int64List()
		BytesList()
			# writer.write(example.SerializeToString())
			
	image:
		decode_jpeg()
		resize()
	
		
			
			
	constant()   # 常量值不可改变，常量的重新赋值相当于是创造新的内存空间
	cast()
	Veriable()
	name_scope()
	
	int64
	bool
	double
	string
	strings:
		regex_full_match()
		join()
		length()
		format()
	
	rank() # tf.rank的作用与numpy的ndim方法相同
	random
		uniform()
		normal()
		truncated_normal()
		
	range()
	linspace()
	
	zeros()
	ones()
	zeros_like()
	eye()
	linalg：
		diag() #对角阵
		inv()
		trace()
		norm()
		det()
		eigvalsh()
		qr()
		svd()
	fill()

	slice(input,begin_vector,size_vector)
	gather(scores,[0,5,9],axis=1)   #抽取每个班级第0个学生，第5个学生，第9个学生的全部成绩
	boolean_mask()
	where()
	scatter_nd()
	
	squeeze()
	transpose()
	reshape()
	expand_dims()
	
	concat()
	stack()
	split()
	add_n()
	
	maximum()
	minimum()
	
	reduce_sum()
	reduce_mean()
	reduce_max()
	reduce_min()
	reduce_prod()
	reduce_all()
	reduce_any()
	
	pow()
	abs()
	
	math；
		cumsum()
		cumprod()
		argmax()
		argmin()
		top_k()
		mod()



	GradientTape():  # 求梯度
		# dy_dx = tape.gradient(y, x)
		# tape.watch([a,b,c]) # 对常量求导，需要增加watch
		# dy_dx,dy_da,dy_db,dy_dc = tape.gradient(y,[x,a,b,c])
		
	
	
	function   # @tf.function() Autograph
	
	Module:   #作为基类被继承，封装autograph为类的形式
		
		class DemoModule(tf.Module):
			pass
	
	
	#GPU
	device("/cpu:0")：
		# with tf.device("/cpu:0"):
			pass
		
	
	config:
		list_physical_devices("GPU")
		experimental:
			set_memory_growth(gpu0, True) # 显存按需使用
			VirtualDeviceConfiguration(memory_limit=4096) #设置GPU显存为固定使⽤量(例如： 4G)
			set_virtual_device_configuration(gpu0, tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096))
	
	distribute:   # 分布式训练
		MirroredStrategy()
		
			# strategy = tf.distribute.MirroredStrategy()  
			# with strategy.scope():
			      model = 
				  model.summary()
				  model = compile_model(model)
			# model.fit()
	
	
	summary:
		create_file_writer()
		trace_on()
		trace_export()
	
	
	saved_model:  （继承tf.Module构建的模型）
		save()
		load()
	
	
	# tensorflow2怀旧版静态图计算
	compat:
		v1:
			Graph()
			placeholder()
			Session()
			
		# tensorflow2怀旧版静态图计算
		import tensorflow as tf
		
		g = tf.compat.v1.Graph()
		with g.as_default():
			x = tf.compat.v1.placeholder(name, shape, dtype)
			y = tf.compat.v1.placeholder(name, shape, dtype)
			z = tf.strings.join([x,y], name="join", separator=" ")
		
		with tf.compat.v1.Session(graph=g) as sess:
			# fetches的结果非常像一个函数的返回值，而feed_dict中的占位符相当于函数的参数列表
			result = sess.run(fetches=z, feed_dict= {x:"hello", y:"world"})



tensorboard:
	notebook:   # tensorboard --logdir ./data/keras_model
		list()
		start()




#实例
tensor:
	numpy()
	assign_add() # 变量的值可以改变，可以通过assign, assign_add等⽅法给变量新赋值
	assign()
	

#实例
tape:
	gradient(y, x)
	watch()


# 
optimizer:
	apply_gradients(grads_and_vars)
	minimize() #相当于是先用tape求gradient，再apply_gradient()
	
	iterations # 用于记录迭代次数


#metrics实例
	update_state(loss)
	update_state(labels, predictions)

	reset_states()


# 实例
dataset；
	shuffle()
	batch()
	repeat()
	shard()
	padded_batch()
	
	map()          # 可设置num_parallel_calls 让数据转换过程多进程执行。
	flat_map()
	interleave()
	filter()   # 过滤
	zip()
	concatenate()
	reduce()
	take(5)
	
	prefetch()     # 让数据准备和参数迭代两个过程相互并行
	interleave()   # 让数据读取过程多线程执行，并将不同来源数据夹在一起
	cache()        # 让数据在第一个epoch后缓存到内存中，仅限于数据集不大的情形
				   # 使用 map转换时，先batch, 然后采用向量化的转换⽅法对每个batch进行转换

# summary实例
writer：
	as_default()


	
#实例
model：
	add()               # 针对Sequential()
	build(input_shape)  # 自定义模型的时，需要build()来初始化变量
	
	variables           # 模型中所有参数变量
		model.layers[0].trainable = False     # 冻结第0层的变量，使其不可训练
		model.trainable_variables
	
	submodules          # 查看模型子模块 （继承tf.Module）
	layers
	name 
	name_scope()
	
	summary()
	compile()
	fit()               # 支持对numpy.array,tf.data.Dataset以及Python generator数据进行训练
	predict()
	predict_classes()
	evaluate()
	predict_on_batch()
	
	reset_metrices()
	model.optimizer.lr.assign(model.optimizer.lr/2.0)  # 重新设置学习率
	
	train_on_batch(x,y)  # 内置方法相对于fit方法更为灵活，可以不通过回调函数，直接在批次上更加精细的控制训练
	test_on_batch(x,y) 
	
	#自定义训练循环
		# 不编译模型
		# 定义loss、metrics、
		# 利用GradientTape、apply_gradients、 update_state 自己写训练循环
	

	save()
	load_model()
		# 保存模型结构及权重 (Keras方式保存)
		model.save('./data/keras_model.h5')
		del model 
		model = tf.keras.models.load_model('./data/keras_model.h5')
		model.evaluate(x_test, y_test)
		
		
		# 保存模型结构与模型参数（tensorflow 方式保存） 具有跨平台便于部署
		model.save('./data/tf_model_savedmodel',save_format="tf")
		model_loaded = tf.keras.models.load_model('./data/tf_model_savedmodel')
		model_loaded.evaluate(x_test,y_test)
		
	
	to_json()
	model_from_json()
		
		# 保存模型结构(Keras方式保存)
		json_str = model.to_json()
		model_json = tf.keras.models.model_from_json(json_str)
		
	save_weights()
	load_weights()
		
		# 保存模型权重（Keras方式保存）
		model.save_weights('./data/keras_model_weight.h5')
		model_json = tf.keras.models.model_from_json(json_str)
		model_json.compile(optimizer, loss, metrics)
		model_json.load_weights('./data/keras_model_weight.h5')
	
		# 保存模型权重（Tensorflow 方式保存）
		model.save_weights('./data/tf_model_weights.ckpt', save_format="tf")
	