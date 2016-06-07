

$('#itype').on('change', function() {
    var itype = $('input[name="itype"]:checked').val();
    var pl = 'Comment Text';

    if (itype == 'rev_id')
        pl = 'Revision ID';

    $('#model-input').attr("placeholder", pl ); 
});


$('#model-submit').click(function () {

        mtype = $('input[name="mtype"]:checked').val();
        itype = $('input[name="itype"]:checked').val();
        idata  = encodeURIComponent($('#model-input').val());
        console.log(itype)

        $.get(
            'api?model=' + mtype + '&input_type=' + itype + '&input=' + idata, // The URL to call
            function(data) { // Success event handler
                console.log(idata)
                if( !('p' in data)) {
                    if (!('error' in data)){
                        $('#model-score').text('ERROR: unknown')
                    }else{
                        $('#model-score').text(data['error'])
                    }
                    
                }else {
                    arr = data['p']
                    res = ''
                    for(var i=0; i < arr.length; i++){
                        res += '<p>' + arr[i][0] + ': ' + arr[i][1]
                    }
                    $('#model-score').empty()

                    if (itype == 'rev_id'){
                        $('#model-score').append('<p><b>Comment: </b></p>')
                        $('#model-score').append('<p>' + data['text'] + '<p>')
                    }
                    $('#model-score').append('<p><b>Results: </b></p>')
                    $('#model-score').append(res)
                }
            }
        );
});

