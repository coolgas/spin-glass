"""Pure early-growth policy; thresholds are provisional, not accuracy bounds."""
import statistics
import math

DECISIONS = (20, 40, 60)


def locally_stable(history, moments, sweep, n):
    """Short-window stability for the disagreement trigger, not final acceptance."""
    if len(history) < 6 or any(s not in moments for s in (sweep-5, sweep)):
        return False
    tail = history[-6:]
    cap = tail[-1]['D']
    if any(r['D'] != cap or r['mixer'] for r in tail):
        return False
    a, b = moments[sweep-5], moments[sweep]
    return (a['D'] == b['D'] == cap and not a['mixer'] and not b['mixer']
            and all(math.isfinite(r['energy']) for r in tail)
            and max(abs(y['energy']-x['energy'])/n for x,y in zip(tail,tail[1:])) < 1e-8
            and all(math.isfinite(x) for x in a['m']+b['m'])
            and max(abs(x-y) for x,y in zip(a['m'],b['m'])) < 1e-4)


def growth_reason(history, moments, cap, sweep, n):
    if sweep not in DECISIONS or len(history) < 5:
        return []
    recent = history[-5:]
    if any(r['D'] != cap or r['mixer'] for r in recent):
        return []
    if sum(r['max_bond'] >= cap for r in recent) < 4:
        return []
    reasons = []
    if statistics.median(r['discarded_weight'] for r in recent) > 1e-6:
        reasons.append('persistent_truncation')
    if sweep >= 40 and len(history) >= 11:
        window = history[-11:]
        if all(r['D'] == cap and not r['mixer'] for r in window):
            changes = [abs(b['energy'] - a['energy']) / n
                       for a, b in zip(window, window[1:])]
            old, new = max(changes[:5]), max(changes[5:])
            if new >= 1e-8 and new >= old / 2:
                reasons.append('energy_stall')
    if sweep >= 40 and all(s in moments for s in (sweep-10, sweep-5, sweep)):
        a, b, c = [moments[s] for s in (sweep-10, sweep-5, sweep)]
        if all(m['D'] == cap and not m['mixer'] for m in (a, b, c)):
            old = max(abs(x-y) for x, y in zip(a['m'], b['m']))
            new = max(abs(x-y) for x, y in zip(b['m'], c['m']))
            if new >= 1e-4 and new >= old / 2:
                reasons.append('magnetization_stall')
    return reasons


def final_window_pass(history, moments, sweep, n):
    if len(history) < 51 or sweep-20 not in moments or sweep not in moments:
        return False
    tail = history[-51:]
    cap = tail[-1]['D']
    if any(r['D'] != cap or r['mixer'] for r in tail):
        return False
    a, b = moments[sweep-20], moments[sweep]
    if a['D'] != cap or b['D'] != cap or a['mixer'] or b['mixer']:
        return False
    if not (all(math.isfinite(r['energy']) for r in tail)
            and all(math.isfinite(x) for x in a['m']+b['m'])):
        return False
    return (max(abs(y['energy']-x['energy'])/n for x, y in zip(tail, tail[1:])) < 1e-8
            and max(abs(x-y) for x, y in zip(a['m'], b['m'])) < 1e-4)
