/**
 * footnote tooltip
 */
const initFootnoteTooltip = () => {
  const fn_refs = document.getElementsByClassName("footnote-ref");
  for (let fn_ref of fn_refs) {
    var parent_ref = document.getElementById(fn_ref.parentElement.id);
    const fn_href = fn_ref["href"];
    const fn_id = fn_href.substr(fn_href.lastIndexOf('#')+1);

    var fn_text = document.getElementById(fn_id).cloneNode(true);
    fn_text.getElementsByClassName("footnote-backref")[0].remove();
    fn_text.innerHTML = fn_text.innerHTML.trim();

    var outer_wrapper = document.createElement("span");
    outer_wrapper.classList.add("footnote-content", "bg-neutral", "dark:bg-neutral-700");
    var inner_wrapper = document.createElement("span");
    inner_wrapper.classList.add("footnote-text", "prose", "dark:prose-invert");

    inner_wrapper.appendChild(fn_text.firstChild);
    outer_wrapper.appendChild(inner_wrapper);
    fn_ref.after(outer_wrapper);
  }
  // $(".footnote-ref").each(function () {
  //   var id = this.attr("href").substr(1),
  //     footnote = $(document.getElementById(id)).clone(),
  //     outer_wrapper = $("<span>", { "class": "fn-content" }),
  //     inner_wrapper = $("<span>", { "class": "fn-text" });
  //   footnote.find(".footnote-return").remove();
  //   this.append(outer_wrapper.append(inner_wrapper.html(footnote.html())));
  // });

  // // fix tooltip position & width
  // var position = function () {
  //   var content = $(".fn-content").removeAttr("style");
  //   if (window.width() < 640)
  //     content.css("width", window.width() / 2);
  //   else
  //     content.css("width", 340); // default value
  //   content.each(function () {
  //     var width = this.children(".fn-text").outerWidth();
  //     this.css({
  //       "width": width,
  //       "margin-left": width / -2
  //     });
  //   });
  // }
  // position();
  // window.resize(position());
}

window.onload = async function() {
  initFootnoteTooltip();
};
