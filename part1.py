import tensorflow as tf
import csv
with open('E:/Final Year/Data Science/training.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')

    targetC = []
    inp = []

    included_cols = list(range(0, 8))
    t_col = [8]

    for row in readCSV:
        t = list(row[i] for i in t_col)
        targetC.append(t)
        i = list(row[i] for i in included_cols)
        inp.append(i)


with open('E:/Final Year/Data Science/testing.csv') as csvfile1:
    readCSV1 = csv.reader(csvfile1, delimiter=',')

    targetC2 = []
    inp2 = []

    included_cols2 = list(range(0, 8))
    t_col2 = [8]

    for row in readCSV1:
        t = list(row[i] for i in t_col2)
        targetC2.append(t)
        i = list(row[i] for i in included_cols2)
        inp2.append(i)

print(inp2)
print(targetC2)

Input=tf.placeholder('float', shape=[None ,8],name="Input")
inputBias = tf.Variable(initial_value=tf.random_normal(shape=[10],stddev=0.4) ,dtype='float', name="input_bias")
weights= tf.Variable(initial_value=tf.random_normal(shape=[8 , 10],stddev=0.4) ,dtype='float' ,name="hidden_weights")

hiddenLayer =tf.matmul(Input,weights)+inputBias
hiddenLayer=tf.sigmoid(hiddenLayer,name='hidden_layer_activation')

weights2= tf.Variable(initial_value=tf.random_normal(shape=[10 , 10],stddev=0.4) ,dtype='float' ,name="hidden_weights2")
inputBias2 = tf.Variable(initial_value=tf.random_normal(shape=[10],stddev=0.4) ,dtype='float', name="input_bias2")

hiddenLayer2 =tf.matmul(hiddenLayer,weights2)+inputBias2
hiddenLayer2=tf.sigmoid(hiddenLayer2,name='hidden_layer_activation')

hiddenBias = tf.Variable(initial_value=tf.random_normal(shape=[1],stddev=0.4) ,dtype='float', name="hidden_bias")

outputWeights=tf.Variable(initial_value=tf.random_normal(shape=[10, 1],stddev=0.4), dtype='float' , name="output_weights")

output= tf.matmul(hiddenLayer2,outputWeights)+hiddenBias
output=tf.sigmoid(output,name='output_layer_activation')

Target=tf.placeholder('float', shape=[None ,1],name="Target")


cost=tf.squared_difference(Target,output)
cost=tf.reduce_mean(cost)
optimizer=tf.train.AdamOptimizer().minimize(cost)

epochs=8000

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(epochs):
        err, _ =sess.run([cost,optimizer], feed_dict={Input: inp,Target:targetC})
        print(i,err)

    for i in range(141):
        test=[inp2[i]]
        predict=sess.run([output],feed_dict={Input:test})
        print(i,targetC2[i],predict[0])