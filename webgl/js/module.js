var gl;             // webgl context
var aCoords;        // Location of the coords attribute variable in the shader program.
var aColor;
var aColorBuffer
var aCoordsBuffer;  // Buffer to hold coords
var uColor;         // Location of the color uniform variable in the shader program.
var uProjection;    // Location of the projection uniform matrix in the shader program
var uModelview;     // Location of the modelview uniform matrix in the shader program
var uNormal;        // Location of the normal.
var uLit;           // Location of the lit
var uNormalMatrix;  

var projection = mat4.create(); // projection matrix
var modelview = mat4.create(); // modelview matrix
var normalMatrix = mat3.create();

var rotator; // A SimpleRotator object to enable rotation by mouse dragging
var loaded_data = [];
var bounce_ids = [];
var loaded_color = [];
var slider = document.getElementById("segmentID");
var slider_text = document.getElementById("segment_id_value")
var segment_id = -1;
var perBounce = false

slider_text.innerHTML = segment_id

slider.oninput = function() {
    // console.log(this.value)
    segment_id = this.value;
    slider_text.innerHTML = this.value;
    draw();
    // segment_id = this.value;
  }

function checkbox_bounce(){
    var checkBox = document.getElementById("perbounce");
    perBounce = checkBox.checked
    draw();
}


function drawPrimitive(primitiveType, color, vertices){
    gl.enableVertexAttribArray(aCoords);
    gl.bindBuffer(gl.ARRAY_BUFFER, aCoordsBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STREAM_DRAW);
    
    gl.enableVertexAttribArray(aColor);
    gl.bindBuffer(gl.ARRAY_BUFFER, aColorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(color), gl.STREAM_DRAW);
    // gl.uniform4fv(uColor, color);
    gl.vertexAttribPointer(aCoords, 3, gl.FLOAT, false, 0, 0);
    gl.vertexAttribPointer(aColor, 1, gl.FLOAT, false, 0, 0);
    gl.drawArrays(primitiveType, 0, vertices.length/3);
}

function draw() {
    
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    mat4.perspective(projection, Math.PI/4, 1, 2, 10);
    gl.uniformMatrix4fv(uProjection, false, projection)

    var modelview = rotator.getViewMatrix();
    gl.uniformMatrix4fv(uModelview, false, modelview);

    mat3.normalFromMat4(normalMatrix, modelview)
    gl.uniformMatrix3fv(uNormalMatrix, false, normalMatrix)

    // gl.uniform1i( uLit, 1);

    // gl.uniform3f(uNormal, 0, 0, 1)
    // drawPrimitive(gl.TRIANGLE_FAN, [1, 0, 0, 1], [-1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1])

    // gl.uniform1i(uLit, 0); 

    gl.lineWidth(5);

    if (loaded_data.length == 0){    
        drawPrimitive(gl.LINES, [1.0, 1.0], [-2, 0, 0, 2, 0, 0]);
        drawPrimitive(gl.LINES, [1.0, 1.0], [-1, 2, 0, 0, 1, 1]);
    }else{
        
        for (l = 0 ; l < loaded_data.length ; l++){
            // cc = [1, 1, 1, brightness];
            // id = l % 3;
            // cc[id] = cc - l * 0.2
            
            if (perBounce & bounce_ids[l] == segment_id){
                cc = loaded_color[l]
                
                drawPrimitive(gl.LINES, cc, loaded_data[l]);
            }
            if (!perBounce & (segment_id == l | segment_id == -1)){   
                // console.log(loaded_color[l].length)
                cc = [...loaded_color[l], ...loaded_color[l]];
                // console.log(cc.length)
                // console.log(cc)
                drawPrimitive(gl.LINES, cc, loaded_data[l]);
            }
        }
    }
        
    
    gl.lineWidth(2);
}

function createProgram(gl, vertexShaderSource, fragmentShaderSource){
    var vsh = gl.createShader(gl.VERTEX_SHADER)
    gl.shaderSource(vsh, vertexShaderSource);
    gl.compileShader(vsh);
    if (! gl.getShaderParameter(vsh, gl.COMPILE_STATUS)){
        throw "Error in vertex shader" + gl.getShaderInfoLog(vsh);
    }
    var fsh = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fsh, fragmentShaderSource);
    gl.compileShader(fsh);
    if (! gl.getShaderParameter(fsh, gl.COMPILE_STATUS)){
        throw "Error in fragment shader" + gl.getShaderInfoLog(fsh);
    }
    var prog = gl.createProgram();
    gl.attachShader(prog, vsh);
    gl.attachShader(prog, fsh);
    gl.linkProgram(prog);
    if ( ! gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        throw "Link error in program: " + gl.getProgramInfoLog(prog);
    }
    return prog;
}

function getTextContent(elementID){
    var element = document.getElementById(elementID);
    var fsource = ""
    var node = element.firstChild;
    var str = ""
    while (node){
        if (node.nodeType == 3) // this is a text node
            str += node.textContent;
        node = node.nextSibling;
    }
    return str;
}

function init(){
    try{
        var canvas = document.getElementById("glcanvas");
        gl = canvas.getContext("webgl", {
            premultipliedAlpha: false,
            alpha: false  // Ask for non-premultiplied alpha
          });
        if (! gl){
            gl = canvas.getContext("experimental-webgl", {
                premultipliedAlpha: false,
                alpha: false  // Ask for non-premultiplied alpha
              });
        }
        if (! gl){
            throw "Could not create WebGL context";
        }
        
        // console.log("file");
        // console.log(data);
        // console.log("data");

        var vertexShaderSource = getTextContent("vshader");
        var fragmentShaderSource = getTextContent("fshader");
        var prog = createProgram(gl, vertexShaderSource, fragmentShaderSource);
        gl.useProgram(prog);
        aCoords = gl.getAttribLocation(prog, "coords");
        aColors = gl.getAttribLocation(prog, "color");
        uModelview = gl.getUniformLocation(prog, "modelview");
        uProjection = gl.getUniformLocation(prog, "projection");
        // uColor = gl.getUniformLocation(prog, "color");
        
        uLit = gl.getUniformLocation(prog, "lit");
        uNormal = gl.getUniformLocation(prog, "normal");
        uNormalMatrix = gl.getUniformLocation(prog, "normalMatrix");
        aCoordsBuffer = gl.createBuffer();
        aColorBuffer = gl.createBuffer();
        gl.enable(gl.DEPTH_TEST);
        gl.enable(gl.CULL_FACE); 
        gl.enable(gl.BLEND);
	    // gl.depthMask(false);
        // gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.blendFunc(gl.ONE, gl.ONE);
        gl.blendEquation(gl.FUNC_ADD);
        rotator = new SimpleRotator(canvas, draw);
        rotator.setView([2, 2, 5], [0, 1, 0], 6);
    }
    catch(e) {
        document.getElementById("message").innerHTML = "Could not initialize WebGL: " + e;
        return;
    }
    draw();
}
