
function talk() {
  var user = document.getElementById("userBox").value;
  var url = '/send?name='+user;
  var req = new XMLHttpRequest();
  var t;
  var User="User: ".bold()
  var Visi="Vision Bot: ".bold()
  req.onreadystatechange = function() {
  if (this.readyState == 4 && this.status == 200) {
    t=req.responseText;        
    document.getElementById("userBox").value = "";
    document.getElementById("chatLog").innerHTML += User+user+"<br>";
    document.getElementById("chatLog").innerHTML += Visi+t+"<br>";
    }
  };
  req.open("GET", url, true);
  req.send();
} 