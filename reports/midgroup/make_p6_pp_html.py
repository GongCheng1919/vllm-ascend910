#!/usr/bin/env python3
"""把 reports/midgroup/P6_pp.md 转成自包含 HTML（内嵌 base64 图表 PNG）。

表格/正文全部程序化转换（与 md 逐字一致，无手抄错误）；
图表插入位置：fig1 -> §4 高并发结尾，fig2 -> §5.1 解码结尾，fig3 -> §5.2 prefill 结尾。
输出：reports/midgroup/P6_pp.html
"""
import base64
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(HERE, "P6_pp.md")
FIGDIR = os.path.join(HERE, "figs")
OUT_PATH = os.path.join(HERE, "P6_pp.html")

FIG_SLOTS = [
    # (唯一的锚文本行, 图文件名, 说明)
    ("## 5. 长上下文", "figs/p6_pp_longctx_decode.png", ""),  # decode 图插到 5.2 之前
    ("### 5.2 Prefill", "", ""),
]
# 图片插入位：前缀匹配标题行，插在该标题**之前**（图属于上一节的内容）
INSERT_BEFORE = [
    ("## 5. 长上下文", "figs/p6_pp_highconcurrency.png",
     "图1 高并发吞吐 vs batch（PP=4 / 64 层真实模型；左：绝对 tok/s，右：对 BF16 加速比）"),
    ("### 5.2 Prefill", "figs/p6_pp_longctx_decode.png",
     "图2 长上下文 · 解码吞吐（PP=4 / 64 层，前缀缓存；batch 1 与 8）"),
    ("## 6. 部署建议", "figs/p6_pp_longctx_prefill.png",
     "图3 长上下文 · prefill 延迟（PP=4 / 64 层，全新 prompt，越低越好）"),
]

# ---------------------------------------------------------------- md -> html
def inline(t):
    """行内标记：**bold**、`code`、*斜体*（保守处理）。"""
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
    return t


def lookup(sym):
    """按 symbol 从 CSV/表格源中取数值 —— 本函数预留：数字直接来自 md，不另查表。"""
    return sym


def md_table(rows):
    """rows: 表格的正文行列表（不含分隔符行）。返回 (<table>, 需要跳过的行数)。"""
    # 首行是表头；第二行是 |---| 分隔（跳过）
    header = [c.strip() for c in rows[0].strip().strip("|").split("|")]
    body = []
    consumed = 1
    if consumed < len(rows) and re.match(r"^\s*\|?[\s:|-]+\|?\s*$", rows[consumed]):
        consumed += 1  # 跳过分隔符行
    for r in rows[consumed:]:
        if not r.strip().startswith("|") or not r.strip().endswith("|") or re.match(r"^\s*\|[\s|:-]+\|\s*$", r):
            break
        cells = [c.strip() for c in r.strip().strip("|").split("|")]
        # 补齐单元格数量（md 省略尾列时）
        cells += [""] * (len(header) - len(cells))
        body.append(cells)
        consumed += 1
    html = ['<div class="tblwrap"><table>']
    html.append("<thead><tr>" + "".join(f"<th>{inline(h)}</th>" for h in header) + "</tr></thead>")
    html.append("<tbody>")
    for cells in body:
        html.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells[:len(header)]) + "</tr>")
    html.append("</tbody></table></div>")
    return "".join(html), consumed


def md_toc(md):
    """收集标题，生成目录。mkdocs 风格锚点。"""
    toc, anchors = [], {}
    for line in md.splitlines():
        m = re.match(r"^(#{2,4})\s+(.+)$", line)
        if not m:
            continue
        lvl = len(m.group(1))
        txt = re.sub(r"\*\*(.+?)\*\*", r"\1", m.group(2)).strip()
        slug = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", txt).strip("-").lower()
        if slug in anchors:
            anchors[slug] += 1
            slug = f"{slug}-{anchors[slug]}"
        else:
            anchors[slug] = 0
        if lvl <= 3:
            toc.append((lvl, txt, slug))
    return toc


def convert():
    with open(MD_PATH, encoding="utf-8") as f:
        md = f.read()
    lines = md.splitlines()
    html_parts, i = [], 0
    in_code = False
    code_buf = []
    toc = md_toc(md)

    def emit_code():
        nonlocal code_buf
        if code_buf:
            html_parts.append('<pre class="code">' + "\n".join(code_buf) + "</pre>")
            code_buf = []

    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            if not in_code:
                in_code = True
            else:
                in_code = False
                emit_code()
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue
        # 标题
        m = re.match(r"^(#+)\s+(.+)$", line)
        if m:
            emit_code()
            # 图插在标题之前（属于上一节内容）
            for prefix, img, cap in INSERT_BEFORE:
                if line.strip().startswith(prefix):
                    b64 = base64.b64encode(open(os.path.join(HERE, img), "rb").read()).decode()
                    html_parts.append(f'<figure><img src="data:image/png;base64,{b64}" alt="{cap}"/>'
                                      f'<figcaption>{cap}</figcaption></figure>')
            lvl = len(m.group(1))
            txt = re.sub(r"\*\*(.+?)\*\*", r"\1", m.group(2)).strip()
            slug = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", txt).strip("-").lower()
            tag = "h2" if lvl == 2 else ("h3" if lvl == 3 else ("h4" if lvl == 4 else "h1"))
            cls = "sec" if lvl == 2 else ""
            html_parts.append(f'<{tag} id="{slug}" class="{cls}">{inline(m.group(2))}</{tag}>')
            i += 1
            continue
        # 表格
        if line.strip().startswith("|"):
            emit_code()
            tbl, consumed = md_table(lines[i:])
            html_parts.append(tbl)
            i += consumed
            continue
        # 引用块
        if line.strip().startswith(">"):
            emit_code()
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(inline(lines[i].strip()[1:].strip()))
                i += 1
            html_parts.append("<blockquote>" + "<br/>".join(buf) + "</blockquote>")
            continue
        # 列表
        if re.match(r"^[-*]\s+", line) or re.match(r"^\d+\.\s+", line):
            emit_code()
            ordered = bool(re.match(r"^\d+\.\s+", line))
            items, tag = [], "ol" if ordered else "ul"
            while i < len(lines) and (re.match(r"^[-*]\s+", lines[i]) or re.match(r"^\d+\.\s+", lines[i])):
                item_txt = re.sub(r"^[-*]\s+|^\d+\.\s+", "", lines[i])
                items.append(f"<li>{inline(item_txt)}</li>")
                i += 1
            html_parts.append(f"<{tag}>" + "".join(items) + f"</{tag}>")
            continue
        # 段落
        if line.strip():
            emit_code()
            buf = [inline(line.strip())]
            while i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].strip().startswith(("#", "|", ">", "-", "*")) and not re.match(r"^\d+\.\s+", lines[i + 1]):
                i += 1
                buf.append(inline(lines[i].strip()))
            html_parts.append("<p>" + " ".join(buf) + "</p>")
        i += 1
    emit_code()
    return "".join(html_parts), toc


def build_toc_html(toc):
    items = []
    for lvl, txt, slug in toc:
        pad = "  " * (lvl - 2)
        items.append(f'{pad}<a href="#{slug}" class="toc-l{lvl}">{txt}</a>')
    return "\n".join(items)


def main():
    body, toc = convert()
    toc_html = build_toc_html(toc)
    css = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#1a2233;--muted:#5b6b81;--line:#e3e7ee;
--accent:#b31e30;--accent2:#0f5ea8;--code-bg:#f0f2f6;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.7 -apple-system,"Segoe UI","PingFang SC","Microsoft YaHei","Noto Sans CJK SC",sans-serif;}
.wrap{max-width:1060px;margin:0 auto;padding:32px 28px 80px;}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:26px 34px;margin:14px 0;box-shadow:0 1px 3px rgba(20,30,60,.06);}
h1{font-size:26px;line-height:1.35;margin:.2em 0 .3em;color:var(--ink);}
h1 .doc{color:var(--muted);font-weight:400;font-size:15px;display:block;margin-top:6px;}
h2.sec{font-size:20px;margin:36px 0 8px;padding:10px 0 8px;border-bottom:3px solid var(--line);color:var(--ink);}
h2.sec::before{content:"";display:inline-block;width:6px;height:1em;background:var(--accent);
margin-right:10px;border-radius:2px;vertical-align:-2px;}
h3{font-size:17px;margin:26px 0 6px;color:var(--ink);}
h4{font-size:15px;margin:18px 0 4px;color:var(--ink);}
p{margin:10px 0;}
a{color:var(--accent2);text-decoration:none;}
a:hover{text-decoration:underline;}
code{background:var(--code-bg);border-radius:4px;padding:1px 5px;font-size:.9em;
font-family:ui-monospace,SFMono-Regular,Consolas,monospace;}
pre.code{background:#0f172a;color:#e2e8f0;border-radius:8px;padding:14px 16px;
overflow-x:auto;font-size:13px;line-height:1.55;}
blockquote{margin:12px 0;padding:10px 16px;border-left:4px solid var(--accent2);
background:#f0f6fc;border-radius:0 8px 8px 0;color:var(--muted);
font-size:.93em;}
ul,ol{padding-left:24px;margin:8px 0;}
li{margin:4px 0;}
strong,b{color:var(--ink);}
.tblwrap{overflow-x:auto;margin:14px 0;}
table{border-collapse:collapse;width:100%;font-size:14px;background:#fff;}
th{background:#f0f3f8;text-align:left;padding:8px 12px;border:1px solid var(--line);
font-weight:600;white-space:nowrap;}
td{padding:7px 12px;border:1px solid var(--line);vertical-align:top;}
tbody tr:nth-child(even){background:#fafbfd;}
figure{margin:22px 0;text-align:center;}
figure img{max-width:100%;border:1px solid var(--line);border-radius:8px;box-shadow:0 2px 8px rgba(20,30,60,.08);}
figcaption{color:var(--muted);font-size:13px;margin-top:6px;}
.toc{columns:1;column-gap:24px;}
.toc a{display:block;padding:2px 0;color:var(--muted);font-size:14px;}
.toc a.toc-l2{font-weight:600;color:var(--ink);margin-top:6px;}
.toc a.toc-l3{padding-left:18px;}
.toc a.toc-l4{padding-left:36px;}
.nav{background:#0f172a;color:#e2e8f0;padding:14px 0;}
.nav .wrap{display:flex;gap:16px;align-items:baseline;padding-top:0;padding-bottom:0;}
.nav .brand{font-weight:700;color:#fff;}
.nav a{color:#9fb3c8;font-size:13px;}
.banner{border-left:6px solid var(--accent);background:#fdf1f2;
border-radius:0 8px 8px 0;padding:12px 18px;margin:12px 0;font-size:.95em;}
small{color:var(--muted);}
@media(max-width:720px){.card{padding:18px 16px;}body{font-size:15px;}}
"""
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>P6 — mid-group W4A8 在 vLLM 流水并行（PP）下的部署报告</title>
<style>{css}</style>
</head>
<body>
<nav class="nav"><div class="wrap">
<span class="brand">P6 · PP 部署报告</span>
<a href="#toc">目录</a>
<a href="P6_pp.md">Markdown 源</a>
</div></nav>
<div class="wrap">
<div class="card">
<h1>P6 — mid-group W4A8 在 vLLM 流水并行（PP）下的部署报告
<span class="doc">2026-08-17 · Ascend 910B4 ×4 · CANN 8.5.0 · vLLM v0.13.0 + vllm-ascend · torch 2.8.0 · QwQ-32B · 数据 <code>int4_cube_lab/results/p6_pp_*.csv</code></span></h1>
<div class="banner"><b>一句话结论：</b>PP 是今天就能部署的那条并行，我们的 W4A8 在它下面一点没掉——真实 64 层
QwQ-32B 跑 PP=4，解码对 BF16 <b>2.0–2.2×</b>，且<b>长上下文下最稳</b>（8192 上下文仍 2.05×）。
但有两条硬边界：<b>PP 给容量不给延迟</b>；<b>prefill 我们是输的</b>（W4A8 只有我们 W8A8 的 0.64–0.69×），
<b>W4A8 是解码方案，不是 prefill 方案</b>。</div>
</div>
<div class="card" id="toc"><h2>目录</h2><div class="toc">{toc_html}</div></div>
{body}
<footer style="margin-top:40px;color:var(--muted);font-size:13px;text-align:center">
P6 PP 部署报告 · 图表由 <code>make_p6_pp_figs.py</code> 从 CSV 数据生成 · 报告由 <code>P6_pp.md</code> 程序化转换
</footer>
</div>
</body>
</html>"""
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"written: {OUT_PATH} ({os.path.getsize(OUT_PATH)/1024:.0f} KB)")


if __name__ == "__main__":
    main()