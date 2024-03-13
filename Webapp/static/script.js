document.addEventListener('DOMContentLoaded', () => {
    const recordButton = document.getElementById('recordButton');
    const transcriptionTextArea = document.getElementById('transcription');

    recordButton.addEventListener('click', () => {
        recordButton.disabled = true;
        transcriptionTextArea.value = 'Recording...';

        // Make POST request to Flask server to start recording
        fetch('/record', {
            method: 'POST'
        })
        .then(response => response.text())
        .then(transcription => {
            transcriptionTextArea.value = transcription;
            recordButton.disabled = false;
        })
        .catch(error => {
            console.error('Error:', error);
            transcriptionTextArea.value = 'Error occurred. Please try again.';
            recordButton.disabled = false;
        });
    });
});
