

$('#itype').on('change', function() {
   $('#model-input').attr("placeholder", $('input[name="itype"]:checked').val()); 
});


$('#model-submit').click(function () {

        mtype = $('input[name="mtype"]:checked').val()
        itype = $('input[name="itype"]:checked').val()
        idata  = encodeURIComponent($('#model-input').val());

        if (itype == 'Comment Text'){

            $.get(
                'api?model=' + mtype + '&text=' + idata, // The URL to call
                function(data) { // Success event handler
                    if( !('p' in data)) {
                        $('#model-score').text('ERROR')
                    }else {
                        arr = data['p']
                        res = ''
                        for(var i=0; i < arr.length; i++){
                            res += '<p>' + arr[i][0] + ': ' + arr[i][1]
                        }
                        $('#model-score').empty()
                        $('#model-score').append('<p><b>Results: </b></p>')
                        $('#model-score').append(res)
                    }
                }
            );

        }else{

            $.get(
                'api?model=' + mtype + '&rev_id=' + idata, // The URL to call
                function(data) { // Success event handler
                    if( !('p' in data)) {
                        $('#model-score').text('ERROR')
                    }else {
                        $('#model-score').empty()
                        $('#model-score').append('<p><b>Comment: </b></p>')
                        $('#model-score').append('<p>' + data['text'] + '<p>')
                        arr = data['p']
                        res = ''
                        for(var i=0; i < arr.length; i++){
                            res += '<p>' + arr[i][0] + ': ' + arr[i][1]
                        }
                        $('#model-score').append('<p><b>Results: </b></p>')
                        $('#model-score').append(res)
                    }
                }
            );
        }


});

