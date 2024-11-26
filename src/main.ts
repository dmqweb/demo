import './style.css'
// 使用tensorflow.js训练模型
import * as tf from "@tensorflow/tfjs/dist/tf";
import traningJSON from "./training.json";
import testJSON from "./test.json";

// 训练数据,测试数据,输出数值,model模型
let trainingData, testingData, outputData, model;
let training = true; //正在训练
let predictButton = document.getElementsByClassName("predict")[0];

const init = async () => { //首先初始化操作
  splitData(); //分割数据
  createModel(); //创建模型
  await trainData(); //等待训练数据
  if (!training) { //如果训练结束,获取数据,进行预测
    predictButton.disabled = false;
    predictButton.onclick = () => {
      const inputData = getInputData();
      predict(inputData);
    };
  }
};

const splitData = () => {
  // tensor2d用于创建一个二维的张量,表示数据集、权重矩阵、图像等
  trainingData = tf.tensor2d(
    // 接收一个数据矩阵,和张量形状、数据类型
    traningJSON.map(item => [
      item.sepal_length,
      item.sepal_width,
      item.petal_length,
      item.petal_width
    ]),
    [130, 4]
  );
  testingData = tf.tensor2d(
    testJSON.map(item => [
      item.sepal_length,
      item.sepal_width,
      item.petal_length,
      item.petal_width
    ]),
    [14, 4]
  );
  // 输出数据
  outputData = tf.tensor2d(
    traningJSON.map(item => [
      item.species === "setosa" ? 1 : 0,
      item.species === "virginica" ? 1 : 0,
      item.species === "versicolor" ? 1 : 0
    ]),
    [130, 3]
  );
};
// 创建模型
const createModel = () => {
  // 创建一个Sequential模型(构建简单线性堆叠的神经网络层结构),该模型按照被添加的顺序,逐层应用各层进行向前传播
  model = tf.sequential();
  // tf.layers.dense用于创建一个全连接层,通过输入数据通过一个线性变换加上一个激活函数来引入非线性,使得神经网路能够学习复杂的模式
  //输入维度\层的激活函数(sigmoid用于将输出压缩到0和1之间,用于二元分类)\层中的神经元数量
  model.add(
    tf.layers.dense({ inputShape: 4, activation: "sigmoid", units: 10 })
  );
  // 添加第二个全连接层,softmax函数用于将输出转换为概率分布
  model.add(
    tf.layers.dense({
      inputShape: 10,
      units: 3,
      activation: "softmax"
    })
  );
  // 编译模型
  model.compile({
    loss: "categoricalCrossentropy",
    optimizer: tf.train.adam()
  });
};
// 训练数据
const trainData = async () => {
  let numSteps = 15;
  let trainingStepsDiv = document.getElementsByClassName("training-steps")[0];
  for (let i = 0; i < numSteps; i++) { //等待模型训练十五次
    let res = await model.fit(trainingData, outputData, { epochs: 40 });
    console.log(res.history, '训练的历史记录');
    trainingStepsDiv.innerHTML = `训练 步骤: ${i}/${numSteps - 1}, 丢失: ${res.history.loss[0]
      }`;
    if (i === numSteps - 1) {
      training = false;
    }
  }
};
// 预测数据
const predict = async inputData => {
  for (let [key, value] of Object.entries(inputData)) {
    inputData[key] = parseFloat(value);
  }
  inputData = [inputData];
  // 对输入进行新的预测
  let newDataTensor = tf.tensor2d(
    inputData.map(item => [
      item.sepal_length,
      item.sepal_width,
      item.petal_length,
      item.petal_width
    ]),
    [1, 4]
  );
  let prediction = model.predict(newDataTensor);
  displayPrediction(prediction);
};
const getInputData = () => {
  let sepalLength = document.getElementsByName("sepal-length")[0].value;
  let sepalWidth = document.getElementsByName("sepal-width")[0].value;
  let petalLength = document.getElementsByName("petal-length")[0].value;
  let petalWidth = document.getElementsByName("petal-width")[0].value;
  return {
    sepal_length: sepalLength,
    sepal_width: sepalWidth,
    petal_length: petalLength,
    petal_width: petalWidth
  };
};
// 对预测进行分类
const displayPrediction = prediction => {
  let predictionDiv = document.getElementsByClassName("prediction")[0];
  let predictionSection = document.getElementsByClassName(
    "prediction-block"
  )[0];
  // 获取到每一个分类的概率分布
  let maxProbability = Math.max(...prediction.dataSync());
  let predictionIndex = prediction.dataSync().indexOf(maxProbability);
  let irisPrediction;
  switch (predictionIndex) {
    case 0:
      irisPrediction = "Setosa";
      break;
    case 1:
      irisPrediction = "Virginica";
      break;
    case 2:
      irisPrediction = "Versicolor";
      break;
    default:
      irisPrediction = "";
      break;
  }
  predictionDiv.innerHTML = irisPrediction;
  predictionSection.style.display = "block";
};

init();
