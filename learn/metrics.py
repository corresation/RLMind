import sys


class Report:
    def __init__(self):
        self.data = {}
        self.avgs = {}  # key: [total, count]

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, val):
        self.data[key] = val

    def addAvg(self, key, val):
        avg = self.avgs.setdefault(key, [0.0, 0])
        avg[0] += float(val)
        avg[1] += 1

    def avg(self, key, values):
        if values.size:
            self.addAvg(key, values.mean())

    def finish(self):
        for key, (total, count) in self.avgs.items():
            self.data[key] = total / count
        self.avgs = {}

    # Rows starting with "-" are indented, empty rows are spacers, missing values are skipped
    def display(self, rows):
        lines = ["\n" * 8 + "=" * 40]
        for row in rows:
            if not row:
                lines.append("")
                continue
            name = row.lstrip("-")
            indent = len(row) - len(name)
            if name not in self.data:
                continue
            prefix = (" " * (indent - 1) * 3 + " - ") if indent else ""
            lines.append(f"{prefix}{name}: {numToStr(self.data[name])}")
        print("\n".join(lines) + "\n" * 4, flush=True)



def numToStr(val):
    mag = abs(val)
    if (0 < mag < 1e-3) or mag >= 1e11:
        return f"{val:.6e}"
    if val == int(val):
        return f"{int(val):,}"
    return f"{val:,.4f}"


class Wandb:
    def __init__(self, project, group, name, runId):
        try:
            import wandb
        except ImportError:
            sys.exit("wandb isn't installed (pip install wandb), or turn off cfg.sendMetrics")
        self.run = wandb.init(project=project, group=group, name=name, id=runId or None, resume="allow")
        self.runId = self.run.id

    def send(self, report):
        self.run.log(report.data)

    # Without wandb's end-of-run printout
    def finish(self):
        self.run._settings.silent = True
        self.run.finish()
