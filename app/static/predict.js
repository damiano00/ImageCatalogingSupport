let imageLoaded = false;
$("#image-selector").change(function () {
	imageLoaded = false;
	let reader = new FileReader();
	reader.onload = function () {
		let dataURL = reader.result;
		$("#selectedImage").attr("src", dataURL);
		removeHighlights();
		imageLoaded = true;
	}
	
	let file = $("#image-selector").prop('files')[0];
	reader.readAsDataURL(file);
});

function showProgress(percentage) {
	var pct = Math.floor(percentage*100.0);
	$('.progress-bar').html(`Caricamento modello (${pct}%)`);
	console.log(`${pct}% caricato`);
}

let model;
//let is_new_od_model;
let modelLoaded = false;
$( document ).ready(async function () {
	modelLoaded = false;
	$('.progress-bar').html("Caricamento Modello");
	$('.progress-bar').show();
    console.log( "Loading model..." );
	model = await tf.loadGraphModel('improvedModel/model.json', {onProgress: showProgress});
//	is_new_od_model = model.inputs.length == 3;
	console.log( "Modello caricato." );
	console.log("Modello ", model);
	$('.progress-bar').hide();
	modelLoaded = true;
});

function _logistic(x) {
	if (x > 0) {
	    return (1 / (1 + Math.exp(-x)));
	} else {
	    const e = Math.exp(x);
	    return e / (1 + e);
	}
}

async function loadImage(onProgress) {
	console.log( "Pre-processing image..." );
	await $('.progress-bar').html("Pre-elaborazione immagine").promise();
	
	const pixels = $('#selectedImage').get(0);

	
	// Pre-process the image
	const input_size = model.inputs[0].shape[1];

	// 4. TODO - Make Detections
	const img = tf.browser.fromPixels(pixels);
	const resized = tf.image.resizeBilinear(img, [640, 640]);
	//640,480
	const casted = resized.cast('int32');
	//int32
	const expanded = casted.expandDims(0);
		
	return expanded;



	//let image = tf.browser.fromPixels(pixels, 3);
	//image = tf.image.resizeBilinear(image.expandDims().toFloat(), [input_size, input_size]);
	//if (is_new_od_model) {
	//	console.log( "Object Detection Model V2 detected." );
	//	image = is_new_od_model ? image : image.reverse(-1); // RGB->BGR for old models
	//}

	//return image;
}

const ANCHORS = [0.573, 0.677, 1.87, 2.06, 3.34, 5.47, 7.88, 3.53, 9.77, 9.17];
const NEW_OD_OUTPUT_TENSORS = ['detected_boxes', 'detected_scores', 'detected_classes'];
async function predictLogos(inputs) {
	console.log( "Running predictions..." );

	await $('.progress-bar').html("Esecuzione").promise();
	const outputs = await model.executeAsync(inputs);//, is_new_od_model ? NEW_OD_OUTPUT_TENSORS : null);
	const arrays = !Array.isArray(outputs) ? outputs.array() : Promise.all(outputs.map(t => t.array()));
	let predictions = await arrays;

	return predictions;
}

var children = [];
function removeHighlights() {
	for (let i = 0; i < children.length; i++) {
		imageOverlay.removeChild(children[i]);
	}
	children = [];
}
async function highlightResults(predictions) {
	console.log( "Highlighting results..." );
	await $('.progress-bar').html("Mostrando i risultati").promise();

	console.log("Selected image width: ", selectedImage.width)
			console.log("Selected image heigth: ", selectedImage.width)

	removeHighlights();
	console.log(predictions);
	console.log(predictions[1][0].length);
	for (let n = 0; n < predictions[2][0].length; n++) { //model v2 -> 4
		// Check scores
		if (predictions[2][0][n] > 0.45) {//0.66 //model v2-> 4
			const p = document.createElement('p');
			p.innerText = TARGET_CLASSES[predictions[5][0][n]]  + ': ' //model v2 -> 6
				+ Math.round(parseFloat(predictions[2][0][n]) * 100)  //model v2 -> 4
				+ '%';
			console.log("Selected image width: ", selectedImage.width)
			console.log("Selected image heigth: ", selectedImage.height)

			bboxTop = (predictions[1][0][n][0] * selectedImage.height) -10; //model v2 -> 0
			bboxLeft = (predictions[1][0][n][1] * selectedImage.width) + 10;
			bboxHeight = (predictions[1][0][n][2] * selectedImage.height) - bboxTop -10;
			bboxWidth = (predictions[1][0][n][3] * selectedImage.width) - bboxLeft + 10;

			p.style = 'margin-left: ' + bboxLeft + 'px; margin-top: '
				+ (bboxTop - 10) + 'px; width: ' 
				+ bboxWidth + 'px; top: 0; left: 0;';
			const highlighter = document.createElement('div');
			highlighter.setAttribute('class', 'highlighter');
			highlighter.style = 'left: ' + bboxLeft + 'px; top: '
				+ bboxTop + 'px; width: ' 
				+ bboxWidth + 'px; height: '
				+ bboxHeight + 'px;';
			imageOverlay.appendChild(highlighter);
			imageOverlay.appendChild(p);
			children.push(highlighter);
			children.push(p);
		}
	}
}

$("#predict-button").click(async function () {
	if (!modelLoaded) { alert("Il modello deve prima essere caricato"); return; }
	if (!imageLoaded) { alert("Per favore, seleziona prima un'immagine"); return; }
	$('.progress-bar').html("Inizio esecuzione");
	$('.progress-bar').show();

	const image = await loadImage();
	const predictions = await predictLogos(image);
	await highlightResults(predictions);

	$('.progress-bar').hide();
});
