

$('#insult-text-input').click(function () {
    $('#insult-text-score').empty()
    $('#insult-text-score').append('Insult Probability: ')
});


$('#insult-text-submit').click(function () {
    comment = encodeURIComponent($('#insult-text-input').val());
    $.get(
        'api?model=insult&text=' + comment, // The URL to call
        function(data) { // Success event handler
            if( !('p' in data)) {
                $('#insult-text-score').text('ERROR')
            }else {
                html = 'Insult Probability: ' + '<span id="text-prob">' + data['p'] + '<\span>'
                $('#insult-text-score').html(html)
                $('#text-prob').css('color', 'green')
            }
        }
    );
});



$('#insult-id-input').click(function () {
    $('#insult-id-score').empty()
    $('#insult-id-score').text('Insult Probability: ')
});

$('#insult-id-submit').click(function () {
    rev_id = encodeURIComponent($('#insult-id-input').val());
    $.get(
        'api?model=insult&rev_id=' + rev_id, // The URL to call
        function(data) { // Success event handler
            if( !('p' in data)) {
                $('#insult-id-score').text('ERROR')
            }else {
                $('#insult-id-score').empty()
                $('#insult-id-score').append('<p><b>Comment: </b></p>')
                $('#insult-id-score').append('<p>' + data['text'] + '<p>')
                $('#insult-id-score').append('<p>Insult Probability: ' + '<span id="text-prob">' + data['p'] + '<\span></p>')
                $('#text-prob').css('color', 'green')
            }
        }
    );
});