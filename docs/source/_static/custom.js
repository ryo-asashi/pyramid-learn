$(() => {
    /* Use wider container for the page content */
    $(".wy-nav-content").each(function () {
        this.style.setProperty("max-width", "none", "important");
    });

    /* List each class property item on a new line */
    if (window.location.pathname.toLocaleLowerCase().indexOf("pythonapi") !== -1) {
        $(".py.property").each(function () {
            this.style.setProperty("display", "inline", "important");
        });
    }

    /* Apply Custom CSS */
    const customCss = `
        .wy-side-nav-search {
            background-color: #0c283b !important;
        }
    `;
    $('<style>').text(customCss).appendTo('body');
});