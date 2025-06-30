var gl;             // webgl context
var aCoords;        // Location of the coords attribute variable in the shader program.
var aCoords_2;
var aColor;
var texcoordLocation;
var aColorBuffer;
var aTexture;
var aCoordsBuffer;  // Buffer to hold coords
var uColor;         // Location of the color uniform variable in the shader program.
var uProjection;    // Location of the projection uniform matrix in the shader program
var uModelview;     // Location of the modelview uniform matrix in the shader program
var uIntensity;
var uSlicePosition;
var uSliceCoord;

var projection = mat4.create(); // projection matrix
var modelview = mat4.create(); // modelview matrix
var normalMatrix = mat3.create();

var rotator; // A SimpleRotator object to enable rotation by mouse dragging
var loaded_data = [];
var bounce_ids = [];
var loaded_color = [];
var slider = document.getElementById("segmentID");
var slider_text = document.getElementById("segment_id_value");
var intensity_slider = document.getElementById("intensityValue");
var intensity_slider_text = document.getElementById("intensity_value_text");

var slice_coord_slider = document.getElementById("slice_coordinate");
var slice_pos_slider = document.getElementById("slice_position");
var slice_coord_slider_text = document.getElementById("slice_coordinate_text");
var slice_pos_slider_text = document.getElementById("slice_position_text");
var slice_intensity_slider = document.getElementById("slice_intensity");
var slice_intensity_text = document.getElementById("slice_intensity_text");

var segment_id = -1;
var source_intensity  = 10;
var perBounce = false;

var drawVoxel = false;
var drawBeam = true;
var energyVoxels;
var vResolution;
var vStepSize;
var vBBox;
var sliceInt = 0; // slices of x, y, or z; use 0, 1, 2 to represent it.
var slicePosition = 0.5; // position of the slice
var slice_position_unormalized;

var beam_prog;
var slice_prog;

slider_text.innerHTML = segment_id;
intensity_slider_text.innerHTML = source_intensity;

slider.oninput = function() {
    // console.log(this.value)
    segment_id = this.value;
    slider_text.innerHTML = this.value;
    draw();
    // segment_id = this.value;
  }

intensity_slider.oninput = function() {
    var p = this.value;
    source_intensity = p;
    intensity_slider_text.innerHTML = source_intensity;
    draw();
}

slice_coord_slider.oninput = function() {
    sliceInt = this.value;
    slice_coord_slider_text.innerHTML = sliceInt;
    //console.log("this value", this.value);
    //console.log(sliceInt);
    draw();
}

slice_pos_slider.oninput = function(){
    slicePosition = (this.value / 100.0);
    slice_pos_slider_text.innerHTML = slicePosition;
    //console.log("this value", this.value);
    //console.log(slicePosition);
    draw();
}

slice_intensity_slider.oninput = function(){
    sliceIntensity = (this.value);
    slice_intensity_text.innerHTML = sliceIntensity;
    // console.log("this value")
    draw();
}

function getSlice_vertices_position(){
    var range = new Float32Array([vBBox[3] - vBBox[0], vBBox[4] - vBBox[1], vBBox[5]- vBBox[2]]);
    var slice_position_ = range[sliceInt] * slicePosition + vBBox[sliceInt];
    if (sliceInt == 0){
        var verts = [
            slice_position_, vBBox[1], vBBox[2],
            slice_position_, vBBox[4], vBBox[2],
            slice_position_, vBBox[1], vBBox[5],
            slice_position_, vBBox[4], vBBox[5],
        ];
        return verts;
    } else if (sliceInt == 1){
        var verts = [
            vBBox[0], slice_position_, vBBox[2],
            vBBox[3], slice_position_, vBBox[2],
            vBBox[0], slice_position_, vBBox[5],
            vBBox[3], slice_position_, vBBox[5],
        ];
        return verts;
    } else if (sliceInt == 2){
        verts = [
            vBBox[0], vBBox[1], slice_position_, 
            vBBox[3], vBBox[1], slice_position_, 
            vBBox[0], vBBox[4], slice_position_, 
            vBBox[3], vBBox[4], slice_position_, 
        ];
        return verts;
    }
}

function getVolumeSliceContent(){
    console.log("vBBox", vBBox);
    var range = new Float32Array([vBBox[3] - vBBox[0], vBBox[4] - vBBox[1], vBBox[5]- vBBox[2]]);
    slice_position_unormalized = range[sliceInt] * slicePosition;

    var newTexture = new Uint8Array(energyVoxels.length);
    for(var i = 0 ; i < energyVoxels.length ; i++){
        newTexture[i] = Math.min(Math.floor(energyVoxels[i] * sliceIntensity), 255);
    }
    // console.log("range ", range);
    // console.log("slicePosition ", slicePosition);
    // console.log("slice position unormalized ", slice_position_unormalized);

    // idx = Math.floor((slice_position_unormalized - vBBox[sliceInt]) / vStepSize[sliceInt]);
    // console.log("idx:", idx);
    // console.log("axis:", sliceInt);

    // if (sliceInt == 0){
    //     num_elements = vResolution[1] * vResolution[2];
    //     slice_texture = new Uint8Array(num_elements);
    //     xid = idx;
    //     for (i = 0 ; i < num_elements ; i++){
    //         yid = i / vResolution[2];
    //         zid = i % vResolution[2];
    //         rid = xid * num_elements + yid * vResolution[2] + zid;
    //         slice_texture[i] = Math.round(energyVoxels[rid] * 255);
    //     }
    //     curent_slice = new Slice(vResolution[1], vResolution[2], slice_texture);
    //     return curent_slice;
    // }else if (sliceInt == 1){
    //     num_elements = vResolution[0] * vResolution[2];
    //     slice_texture = new Uint8Array(num_elements);
    //     yid = idx;
    //     for (i = 0 ; i < num_elements ; i++){
    //         xid = i / vResolution[2];
    //         zid = i % vResolution[2];
    //         rid = xid * vResolution[1] * vResolution[2] + yid * vResolution[2] + zid;
    //         slice_texture[i] = Math.round(energyVoxels[rid] * 255);
    //     }
    //     curent_slice = new Slice(vResolution[0], vResolution[2], slice_texture);
    //     return curent_slice;
    // }else if (sliceInt == 2){
    //     num_elements = vResolution[0] * vResolution[1];
    //     slice_texture = new Uint8Array(num_elements);
    //     zid = idx;
    //     for (i = 0 ; i < num_elements ; i++){
    //         xid = i / vResolution[1];
    //         yid = i % vResolution[1];
    //         rid = xid * vResolution[1] * vResolution[2] + yid * vResolution[2] + zid;
    //         slice_texture[i] = Math.round(energyVoxels[rid] * 255);
    //     }
    //     curent_slice = new Slice(vResolution[0], vResolution[1], slice_texture);
    //     return curent_slice;
    // }   
    // return []; 
    return new Slice3D(vResolution[0], vResolution[1], vResolution[2], newTexture);
}

function checkbox_bounce(){
    var checkBox = document.getElementById("perbounce");
    perBounce = checkBox.checked;
    draw();
}

function checkbox_drawslice(){
    var drawslice_checkbox = document.getElementById("drawslice");
    drawVoxel = drawslice_checkbox.checked;
    draw();
}

function checkbox_drawbeam(){
    var drawbeam_checkbox = document.getElementById("drawbeam");
    drawBeam = drawbeam_checkbox.checked;
    draw();
}


function drawPrimitive(primitiveType, colors, vertices){
    gl.bindBuffer(gl.ARRAY_BUFFER, aCoordsBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STREAM_DRAW);
    gl.vertexAttribPointer(aCoords, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(aCoords);

    
    gl.bindBuffer(gl.ARRAY_BUFFER, aColorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STREAM_DRAW);
    gl.vertexAttribPointer(aColor, 1, gl.FLOAT, true, 0, 0);
    gl.enableVertexAttribArray(aColor);

    gl.drawArrays(primitiveType, 0, vertices.length/3);
}


function drawSlice(texture_data, points){
    gl.bindBuffer(gl.ARRAY_BUFFER, aCoordsBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(points), gl.STREAM_DRAW);
    gl.vertexAttribPointer(aCoords, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(aCoords);

    
    // Create a buffer for texcoords.
    var buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    // Set Texcoords.
    gl.bufferData(
        gl.ARRAY_BUFFER,
        new Float32Array([
          0, 0,
          1, 0,
          0, 1,
          1, 1]),
         gl.STREAM_DRAW);
 
    // We'll supply texcoords as floats.
    gl.vertexAttribPointer(texcoordLocation, 2, gl.FLOAT, true, 0, 0);
    gl.enableVertexAttribArray(texcoordLocation);

    // Create a texture.
    
    gl.bindTexture(gl.TEXTURE_3D, aTexture);
    // Fill the texture with a 1x1 blue pixel.
    // console.log(texture_data);
    // gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, 1, 1, 0, gl.RED, gl.FLOAT, texture_data);
    
    // console.log(ext);
    
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

    console.log("w x h", texture_data.width, texture_data.height);

    gl.texImage3D(gl.TEXTURE_3D, 0, gl.LUMINANCE, texture_data.width, texture_data.height, texture_data.depth, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, texture_data.data);
    gl.generateMipmap(gl.TEXTURE_3D);
    // gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 255, 255]));
            //   new Uint8Array([0, 0, 255, 255]));
    // console.log("drawing picture with texture", texture_data);
    gl.texParameterf(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameterf(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    // gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); 
    // gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, points.length/3);
}

class Slice3D {
    constructor(height, width, depth, data) {
      this.height = height;
      this.width = width;
      this.depth = depth;
      this.data = data;
    }
  }

function render_beam(){
    gl.useProgram(beam_prog);
    aCoords = gl.getAttribLocation(beam_prog, "coords");
    aColor = gl.getAttribLocation(beam_prog, "color");
    uModelview = gl.getUniformLocation(beam_prog, "modelview");
    uProjection = gl.getUniformLocation(beam_prog, "projection");
    uIntensity = gl.getUniformLocation(beam_prog, "intensity");
    aColorBuffer = gl.createBuffer();
    aCoordsBuffer = gl.createBuffer();

    mat4.perspective(projection, Math.PI/4, 1, 2, 10);
    gl.uniformMatrix4fv(uProjection, false, projection);

    var modelview = rotator.getViewMatrix();
    gl.uniformMatrix4fv(uModelview, false, modelview);

}

function render_slice(){
    gl.useProgram(slice_prog);
    aCoords = gl.getAttribLocation(slice_prog, "coords");
    uModelview = gl.getUniformLocation(slice_prog, "modelview");
    uProjection = gl.getUniformLocation(slice_prog, "projection");
    texcoordLocation = gl.getAttribLocation(slice_prog, "a_texcoord");
    uSlicePosition = gl.getUniformLocation(slice_prog, "slice_pos");
    uSliceCoord = gl.getUniformLocation(slice_prog, "slice_coord");
    aTexture = gl.createTexture();
    aCoordsBuffer = gl.createBuffer();

    mat4.perspective(projection, Math.PI/4, 1, 2, 10);
    gl.uniformMatrix4fv(uProjection, false, projection);

    var modelview = rotator.getViewMatrix();
    gl.uniformMatrix4fv(uModelview, false, modelview);
}

function draw() {
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    if(drawVoxel){
        render_slice();
        gl.uniform1f(uSlicePosition, slicePosition);
        gl.uniform1i(uSliceCoord, sliceInt);

        gl.disable(gl.DEPTH_TEST);
        gl.disable(gl.CULL_FACE); 
        gl.enable(gl.BLEND);
    
        gl.blendFunc(gl.ONE, gl.ONE);
        gl.blendEquation(gl.FUNC_ADD);

        if (vStepSize == null){
            var test_pixels = new Uint8Array(16 * 4);
            for(i = 0 ; i < 16 ; i++){
                test_pixels[i * 4] = i * 10;
                test_pixels[i * 4+1] = i * 20;
                test_pixels[i * 4+2] = 255;
                test_pixels[i * 4+3] = 255;
            }
            var test_slice_data = new Slice3D(4, 4, 4, test_pixels);

            drawSlice(test_slice_data, 
                [-1, -1, 0,
                1, -1, 0,
                -1, 1, 0,
                1, 1, 0,
                ], 2, 2);
        }
        else{
            
            // var vertices = [-1, -1, 0,
            //     1, -1, 0,
            //     -1, 1, 0,
            //     1, 1, 0,
            //     ];
            var vertices = getSlice_vertices_position();
            var slice_texture = getVolumeSliceContent();
            drawSlice( slice_texture, vertices);
        }
    }

    if (drawBeam){
        render_beam();
        gl.uniform1f(uIntensity, source_intensity);

        gl.disable(gl.DEPTH_TEST);
        gl.disable(gl.CULL_FACE); 
        gl.enable(gl.BLEND);
    
        gl.blendFunc(gl.ONE, gl.ONE);
        gl.blendEquation(gl.FUNC_ADD);

        gl.lineWidth(5);
        // console.log(loaded_data.length)
        if (loaded_data.length == 0){   
            drawPrimitive(gl.LINES, [1.0, 1.0], [-2, 0, 0, 2, 0, 0]);
            drawPrimitive(gl.LINES, [1.0, 1.0], [0, -2, 0, 0, 2, 0]);
            drawPrimitive(gl.LINES, [1.0, 1.0], [0, 0, -2, 0, 0, 2]);
        }else{
            
            for (l = 0 ; l < loaded_data.length ; l++){
                // console.log("draw beams: ", loaded_data.length)
                
                if (perBounce & bounce_ids[l] == segment_id){
                    cc = [...loaded_color[l], ...loaded_color[l]];
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
        gl.lineWidth(1);
    }
    
}

function createProgram(gl, vertexShaderSource, fragmentShaderSource){
    var vsh = gl.createShader(gl.VERTEX_SHADER);
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
    var fsource = "";
    var node = element.firstChild;
    var str = "";
    while (node){
        if (node.nodeType == 3) // this is a text node
            str += node.textContent;
        node = node.nextSibling;
    }
    return str;
}

function getShaderProgram(vertexshader, fragmentshader){
    var vertexShaderSource = getTextContent(vertexshader);
    var fragmentShaderSource = getTextContent(fragmentshader);
    var prog = createProgram(gl, vertexShaderSource, fragmentShaderSource);
    return prog;
}

function init(){
    try{
        var canvas = document.getElementById("glcanvas");
        gl = canvas.getContext("webgl2", {
            premultipliedAlpha: false,
            alpha: false  // Ask for non-premultiplied alpha
          });
        
        if (! gl){
            console.log("using webgl 1.0")
            gl = canvas.getContext("webgl", {
                premultipliedAlpha: false,
                alpha: false  // Ask for non-premultiplied alpha
              });
        }
        if (! gl){
            throw "Could not create WebGL context";
        }
        const ext = gl.getExtension("EXT_color_buffer_float");
        gl.getExtension('OES_texture_float');
        
        beam_prog = getShaderProgram("vshader", "fshader");
        slice_prog = getShaderProgram("vsliceshader", "fsliceshader");
        
       
        rotator = new SimpleRotator(canvas, draw);
        rotator.setView([2, 2, 5], [0, 1, 0], 6);
    }
    catch(e) {
        document.getElementById("message").innerHTML = "Could not initialize WebGL: " + e;
        return;
    }
    draw();
}
