document.getElementById('BeamInput').addEventListener('change', function (event){
            let file = event.target.files[0];
            let reader = new FileReader();

            reader.onload = function (event) {
                loaded_data = [];
                bounce_ids = [];
                loaded_color = [];
                let arrayBuffer = event.target.result;
                // let array = new Uint8Array(arrayBuffer);
                iterate = 0
                // console.log(iterate)
                // console.log(arrayBuffer.byteLength)
                while (iterate < arrayBuffer.byteLength){
                    
                    // console.log("iterate" + iterate)
                    numer_points = new Uint32Array(arrayBuffer.slice(iterate, iterate + 4))[0];
                    numer_colors = new Uint32Array(arrayBuffer.slice(iterate + 4, iterate + 8))[0];
                    bounceIdx = new Uint32Array(arrayBuffer.slice(iterate + 8, iterate + 12))[0];
                    // console.log("number of points", numer_points)
                    // console.log("number of colors", numer_colors)

                    points = new Float32Array(arrayBuffer.slice(iterate + 12, iterate + 12 + numer_points * 4));
                    colors = new Float32Array(arrayBuffer.slice(iterate + 12 + numer_points * 4, iterate + 12 + numer_points * 4 + numer_colors * 4));
                
                    // let floatarray = new Float32Array(points);
                // console.log(floatarray)

                // let fileSize = arrayBuffer.byteLength;
                // let bytes = [];
                // var number_of_floats = array[0]
                // // console.log(number_of_points)
                // for (let i = 0 ; i < Math.min(20, fileSize) ; i++){
                //     bytes.push(array[i]);
                // }
                    loaded_data.push(points);
                    bounce_ids.push(bounceIdx);
                    loaded_color.push(colors);
                    // break;
                    // console.log(loaded_data)
                    // console.log("float array" + points.length);
                    // console.log("loaded data length"+loaded_data.length);
                    iterate = iterate + 12 + numer_points * 4 + numer_colors * 4;
                }
                
                // loaded_data.join()
                // console.log(loaded_data)
                draw();

                // document.getElementById('fileContents').textContent = 'First 20 bytes of file as ArrayBuffer: '
                //         + bytes.join(', ');
                // console.log('ArrayBuffer:', arrayBuffer);
            };

            reader.readAsArrayBuffer(file);
});

document.getElementById('VoxelInput').addEventListener('change', function (event){
    let file = event.target.files[0];
    let reader = new FileReader();

    reader.onload = function (event) {
        let arrayBuffer = event.target.result;
        
        iterate = 0;
        // console.log("iterate " + iterate);
        vResolution = new Uint32Array(arrayBuffer.slice(iterate, iterate + 12));
        // console.log("vResoltion " + vResolution);
        vStepSize = new Float32Array(arrayBuffer.slice(iterate+12, iterate + 24));
        // console.log("vSteps  " + vStepSize);
        vBBox = new Float32Array(arrayBuffer.slice(iterate+24, iterate + 48))
        // console.log("vBBox " + vBBox);

        var number_elements = vResolution[0] * vResolution[1] * vResolution[2];

        energyVoxels = new Float32Array(arrayBuffer.slice(iterate + 48, iterate + 48 + number_elements * 4));
        cur_voxels = energyVoxels

        draw();
    };

    reader.readAsArrayBuffer(file);
});

// document.getElementById('slidecontainer').addEventListener('change', function (event){
//         segment_id = slider.value;
//         console.log(slider.value)
//         draw();
// });