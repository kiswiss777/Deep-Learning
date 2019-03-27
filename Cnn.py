import tensorflow as  tf
from tensorflow.examples.tutorials.mnist import input_data #데이타셋을 가지고오다
mnist = input_data.read_data_sets('MNIST_data',one_hot=True,reshape=False)#MNIST_data 라는 폴더를 생성하여서 저장 one_hot = 문자를 ->숫자로 바꿔주는거

X = tf.placeholder(tf.float32,shape=[None,28,28,1]) #28*28 이미지를 담을 꺼기때문에 28,28
Y_Label = tf.placeholder(tf.float32,shape=[None,10])  #10개의 아웃풋에 맞게 10개배열생성

Kernel1 = tf.Variable(tf.truncated_normal(shape=[4,4,1,4],stddev=0.1)) # 필터 크기 4*4*1 4개 생성 정규분포 초기화 방법 표준편차가 0.1 평균이 0에가까운 난수를 얻기위함
Bias1 = tf.Variable(tf.truncated_normal(shape=[4],stddev=0.1)) #4개의 필터를 만들었기 때문에 4개생성 맵핑 시키기 위해서
Conv1 = tf.nn.conv2d(X,Kernel1,strides=[1,1,1,1],padding='SAME')+Bias1 #1칸씩 움직이기 때문에 중앙2개 숫자를 1로 지정, #출력크기를 입력과 같게 유지
Activation1 = tf.nn.relu(Conv1) ## relu 함수를 사용해 컨볼루션
Pool1 = tf.nn.max_pool(Activation1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') #overfiting 을 방지하기위해,2칸씩 이동하면서 최댓값을 추출,ksize(weigh,high)

Kernel2 = tf.Variable(tf.truncated_normal(shape=[4,4,4,8],stddev=0.1))
Bias2 = tf.Variable(tf.truncated_normal(shape=[8],stddev=0.1))
Conv2 = tf.nn.conv2d(Pool1,Kernel2,strides=[1,1,1,1],padding='SAME')+Bias2 # 컴퓨터가 학습을 통해 필터를 설정하기때문에 마찬기로 Bias도 조정하면서 한다
Activation2 = tf.nn.relu(Conv2)
Pool2 = tf.nn.max_pool(Activation2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

W1 = tf.Variable(tf.truncated_normal(shape=[8*7*7,10])) #10개의 output을 가지겠다 8*7*7 은 사이즐 거쳐서 나오는것을 나눈것
B1 = tf.Variable(tf.truncated_normal(shape=[10]))
Pool2_flat = tf.reshape(Pool2,[-1,8*7*7])#원래 배치사이즈를 사용하나 잘 모를경우에는 -1로 설정하고 사용하면 된다
OutputLayer = tf.matmul(Pool2_flat,W1)+B1

Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_Label,logits=OutputLayer)) #label= 실제클래스값,logits=출력값
train_step = tf.train.AdamOptimizer(0.005).minimize(Loss) #원하는 것을 찾아간다.

correct_prediction = tf.equal(tf.argmax(OutputLayer,1),tf.argmax(Y_Label,1)) #정확도를 체크하기 위해서
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) #평균값을 나타낸다

with tf.Session() as sess:
    print("Start...")
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        trainingData,Y = mnist.train.next_batch(64)
        sess.run(train_step,feed_dict={X:trainingData,Y_Label:Y})
        if i%100:
            print(sess.run(accuracy,feed_dict={X:mnist.test.images,Y_Label:mnist.test.labels}))
