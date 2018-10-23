(function ($) {
    $(document).ready(() => $.ajax({url: 'alignment.json', dataType: 'json', success: onAlignmentLoaded}));

    let onAlignmentLoaded = function (content) {
        // bootstrap alignment
        let player = $('#player')
        let target = $('#target')
        let alignments = content.alignments

        player.attr('src', 'audio.mp3')
        player.on('timeupdate', alignments, selectWord);
        player.on('seeked', alignments, selectWord);
        player.on('timeupdate', alignments, selectWord);
        player.on('seeked', alignments, selectWord);
        player[0].load();

        alignments.forEach(alignment => align(alignment, player[0], target[0]));

        // add .unaligned clas to all unaligned text nodes
        $('#target').contents()
            .filter(function(){return this.nodeType === 3 && $(this).text() !== ' '})
            .wrap("<span class='unaligned'></span>")

        // enable popovers
        $('[data-toggle="tooltip"]').tooltip({'placement': 'bottom', 'trigger': 'hover'})
    };

    let align = function (alignment_entry, player, target) {
        // align a simple entry from alignments.json: [text::strt, start::float, stop::float] (time in seconds)
        let transcript = alignment_entry[0]
        let alignment = alignment_entry[1];
        let node = createNode(target, transcript, alignment);
        alignment_entry[4] = node
        $(node).click(() => player.currentTime = alignment_entry[2])
    };

    let isTextNodeContaining = function (text) {
        // checks if a given HTML node contains {text}
        return node => {
            let isTextNode = [3, 4].includes(node.nodeType);
            let containsText = node.data.toLowerCase().includes(text.toLowerCase());
            let isAligned = $(node).hasClass('aligned') || $(node.parentElement).hasClass('aligned');
            return isTextNode && containsText && !isAligned
        }
    }

    let createNode = function (target, transcript, alignment) {
        // replaces all occurrences of {text} in target with a <span class='aligned'>{text}</span>
        let textNodes = getTextNodesIn(target);
        let node = textNodes.find(isTextNodeContaining(alignment));
        if (node) {
            let alignmentNode = node.splitText(node.data.toLowerCase().indexOf(alignment.toLowerCase()));
            alignmentNode.splitText(alignment.length)
            let highlightedNode = $('<span></span>')
                .addClass('aligned')
                .attr({'data-toggle': 'tooltip', 'title': transcript});
            $(alignmentNode).replaceWith(highlightedNode);
            highlightedNode.append(alignmentNode);
            return highlightedNode;
        }
    };

    let getTextNodesIn = function (node, includeWhitespaceNodes) {
        // find all text node children in a parent node
        let textNodes = [], nonWhitespaceMatcher = /\S/;
        function getTextNodes(node) {
            if (node.nodeType === 3) {
                if (includeWhitespaceNodes || nonWhitespaceMatcher.test(node.nodeValue)) {
                    textNodes.push(node);
                }
            } else {
                for (var i = 0, len = node.childNodes.length; i < len; ++i) {
                    getTextNodes(node.childNodes[i]);
                }
            }
        }
        getTextNodes(node);
        return textNodes;
    };

    let selectWord = function (e) {
        // selects a word by setting the classes and focus
        let alignments = e.data
        $('.current').removeClass('current')
        alignments.forEach(alignment => {
            if (player.currentTime >= alignment[1] && player.currentTime <= alignment[2] && alignment[3]) {
                let node = alignment[3];
                $(node).addClass('current')
                node.focus();
            }
        })
    };

})($)