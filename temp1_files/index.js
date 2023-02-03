// window.HELP_IMPROVE_VIDEOJS = false;

// var INTERP_BASE = "https://homes.cs.washington.edu/~kpar/nerfies/interpolation/stacked";
// var NUM_INTERP_FRAMES = 240;

// var interp_images = [];
// function preloadInterpolationImages() {
//   for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
//     var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
//     interp_images[i] = new Image();
//     interp_images[i].src = path;
//   }
// }

// function setInterpolationImage(i) {
//   var image = interp_images[i];
//   image.ondragstart = function() { return false; };
//   image.oncontextmenu = function() { return false; };
//   $('#interpolation-image-wrapper').empty().append(image);
// }


// $(document).ready(function() {
//     // Check for click events on the navbar burger icon
//     $(".navbar-burger").click(function() {
//       // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
//       $(".navbar-burger").toggleClass("is-active");
//       $(".navbar-menu").toggleClass("is-active");

//     });

//     var options = {
// 			slidesToScroll: 1,
// 			slidesToShow: 3,
// 			loop: true,
// 			infinite: true,
// 			autoplay: false,
// 			autoplaySpeed: 3000,
//     }

// 		// Initialize all div with carousel class
//     var carousels = bulmaCarousel.attach('.carousel', options);

//     // Loop on each carousel initialized
//     for(var i = 0; i < carousels.length; i++) {
//     	// Add listener to  event
//     	carousels[i].on('before:show', state => {
//     		console.log(state);
//     	});
//     }

//     // Access to bulmaCarousel instance of an element
//     var element = document.querySelector('#my-element');
//     if (element && element.bulmaCarousel) {
//     	// bulmaCarousel instance is available as element.bulmaCarousel
//     	element.bulmaCarousel.on('before-show', function(state) {
//     		console.log(state);
//     	});
//     }

//     /*var player = document.getElementById('interpolation-video');
//     player.addEventListener('loadedmetadata', function() {
//       $('#interpolation-slider').on('input', function(event) {
//         console.log(this.value, player.duration);
//         player.currentTime = player.duration / 100 * this.value;
//       })
//     }, false);*/
//     preloadInterpolationImages();

//     $('#interpolation-slider').on('input', function(event) {
//       setInterpolationImage(this.value);
//     });
//     setInterpolationImage(0);
//     $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

//     bulmaSlider.attach();

// })
// if (document.readyState === "complete") { 
//     alert('cp');
//   (function() {
//     if (document.getElementById('tab1').checked){
//         alert('checked');
// alert( document.getElementsByClassName('vid-item'));
var items = document.getElementsByClassName('vid-item');
items[0].click();
//     }
//   })();      
// }
function selectthis(i) {
//   alert('!!');
    var items = document.getElementsByClassName('vid-item');
    // alert(items);
    [].forEach.call(items, function (j) {
        // j.style.background='#0a0a23';
        j.className='vid-item';});
    var items_c = document.getElementsByClassName('vid-item-c');
    // alert(items);
    [].forEach.call(items_c, function (j) {
        // j.style.background='#0a0a23';
        j.className='vid-item';});
    // i.style.background='rgba(223, 13, 13, 0.9)';
    i.className='vid-item-c';
}
