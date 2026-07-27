"""Strip citation labels ([Aba21]) from bibliography entries."""

import docutils.nodes
from sphinx.transforms.post_transforms import SphinxPostTransform


class StripCitationLabels(SphinxPostTransform):
    default_priority = 6  # runs right after BibliographyTransform (priority 5)

    def run(self):
        for node in self.document.findall(docutils.nodes.citation):
            for label in node.findall(docutils.nodes.label):
                label.parent.remove(label)
            for para in node.findall(docutils.nodes.paragraph):
                para.insert(0, docutils.nodes.Text("\u2022 "))
                break


def setup(app):
    app.add_post_transform(StripCitationLabels)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
