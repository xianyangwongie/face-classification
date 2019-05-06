var MODEL_GENDER;
var IS_MODEL_GENDER_LOADED = false;
var ages = [];
for(i = 0; i < 21; i++){
  ages.push([i])
}

initGender = async () => {
  MODEL_GENDER = await tf.loadModel("./model/gender/model.json");
  console.log("Model Gender Loaded");

  //Warm up network
  MODEL_GENDER.predict(tf.zeros([null, 224, 224, 3]));
  IS_MODEL_GENDER_LOADED = true
  M.toast({html: 'Model Gender Loaded.', displayLength: 1000})
};

initGender();

function predictGender(input) {
  var r = MODEL_GENDER.predict(input);
  var result_gender = r[0].dataSync();
  var result_age = r[1].dataSync();

  var tresult_gender = tf.tensor(result_gender)
  var label_index_gender = tf.argMax(tresult_gender).dataSync()[0]
  var label_percent_gender = result_gender[label_index_gender].toFixed(4) * 100;

  var tresult_age = tf.tensor(result_age)
  var label_index_age = tf.argMax(tresult_age).dataSync()[0]
  var label_percent_age = result_age[label_index_age].toFixed(4) * 100;
  var predicted_age = math.multiply(Array.from(result_age),ages)

  return {
    age: {"result": result_age,"label": parseInt(predicted_age[0]*4.76), "percent": label_percent_age},
    gender: {"result": result_gender, "label": LABEL_GENDER[label_index_gender], "percent": label_percent_gender}
  };
}
