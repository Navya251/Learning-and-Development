function calc() {
  var a = document.querySelector("#value1").value;
  var inta = parseInt(a);
  var b = document.querySelector("#value2").value;
  var intb = parseInt(b); 
  var op = document.querySelector("#operator").value;
  var calculate;

  if(op == "add") {
      calculate = inta + intb;
  } else if(op == "sub") {
      calculate = inta - intb;
  } else if(op == "mul") {
      calculate = inta * intb;
  } else if(op == "div") {
      calculate = inta / intb;
  }

  document.getElementById("result").innerHTML = "Result : " + calculate;
}
