(function ($) {
    $(document).ready(() => $.ajax({url: 'alignment.json', dataType: 'json', success: onAlignmentLoaded}));

    let onAlignmentLoaded = function (content) {
        // bootstrap alignment
        let $player = $('#player')
        let $target = $('#target')
        let alignments = content.alignments

        $.ajax({
            url: 'transcript.txt',
            dataType: 'text',
            success: function (transcript) {
                align($target, transcript, alignments, $player[0])
                // enable popovers
                $('[data-toggle="tooltip"]').tooltip({'placement': 'bottom', 'trigger': 'hover'})
            }
        })

        $player.attr('src', 'audio.mp3')
        $player.on('timeupdate seeked', alignments, highlightAlignment);
        $player[0].load();
    };

    let align = function ($target, transcript, alignments, player) {
        $target.empty()

        let prevEnd = 0
        alignments.forEach(function (alignment) {
            let audioStart = alignment.audio_start;
            let audioEnd = alignment.audio_end;
            let textStart = alignment.text_start;
            let textEnd = alignment.text_end;
            let inference = alignment.transcript;

            if (alignment.text_start > prevEnd && transcript.substring(prevEnd, textStart) !== " ") {
                let unalignedText = transcript.substring(prevEnd, textStart)
                    .replace(/(?:\r\n|\r|\n)/g, '<br/>')
                    .trim();
                $target.append($('<span></span>').html(unalignedText).addClass('unaligned'));
                $target.append(document.createTextNode(' '));
            }

            let alignedText = transcript.substring(textStart, textEnd)
                .replace(/(?:\r\n|\r|\n)/g, '<br/>')
                .trim();
            let tooltipText = inference + ' (' + toHHMMSS(audioStart) + ' - ' + toHHMMSS(audioEnd) + ')';

            let $node = $('<span></span>').html(alignedText)
                .addClass('aligned')
                .attr({'data-toggle': 'tooltip', 'title': tooltipText})
                .click(() => player.currentTime = audioStart);

            alignment.node = $node
            $target.append($node)
            $target.append(document.createTextNode(' '));
            prevEnd = textEnd
        })
    }

    let highlightAlignment = function (e) {
        // selects a word by setting the classes and focus
        let alignments = e.data
        $('.current').removeClass('current')
        alignments.forEach(alignment => {
            if (player.currentTime >= alignment.audio_start && player.currentTime <= alignment.audio_end && alignment.node) {
                let node = alignment.node;
                $(node).addClass('current')
                node.focus();
            }
        })
    };

    let toHHMMSS = function (s) {
        let hours = Math.floor(s / 3600)
        var minutes = Math.floor((s - (hours * 3600)) / 60);
        var seconds = Math.round(s - (hours * 3600) - (minutes * 60));

        if (hours < 10) {
            hours = "0" + hours;
        }
        if (minutes < 10) {
            minutes = "0" + minutes;
        }
        if (seconds < 10) {
            seconds = "0" + seconds;
        }
        return hours + ':' + minutes + ':' + seconds;
    }

})($)