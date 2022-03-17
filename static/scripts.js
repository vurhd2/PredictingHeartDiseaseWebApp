// makes it so a form does not resubmit upon reloading the page
if ( window.history.replaceState ) {
    window.history.replaceState( null, null, window.location.href );
}