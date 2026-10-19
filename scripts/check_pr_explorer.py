"""Browser/Python parity checks for the offline figure (requires Playwright)."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_pr_explorer import metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--figure', type=Path, default=Path('figures/multiclass-pr-explorer.html'))
    p.add_argument('--data', type=Path, default=Path('outputs/pr_explorer'))
    p.add_argument('--chrome', default='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
    a = p.parse_args()
    with np.load(a.data / 'validation_predictions.npz') as data:
        logits = data['logits'].astype(float)
        labels = data['labels'].astype(int)
    with np.load(a.data / 'test_predictions.npz') as data:
        test_logits = (np.log(data['probabilities'].astype(float)) if 'probabilities' in data
                       else data['logits'].astype(float))
        test_labels = data['labels'].astype(int)
    errors, requests, checked = [], [], []
    with sync_playwright() as browser_api:
        browser = browser_api.chromium.launch(executable_path=a.chrome, headless=True)
        page = browser.new_page(viewport={'width': 1500, 'height': 1100}, device_scale_factor=1)
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.route('http://**/*', lambda route: (requests.append(route.request.url), route.abort()))
        page.route('https://**/*', lambda route: (requests.append(route.request.url), route.abort()))
        page.goto(a.figure.resolve().as_uri())
        page.wait_for_function('window.prExplorer && window.prExplorer.state.ready')

        def verify(biases, name):
            expected = metrics(labels, (logits + biases).argmax(1))
            flat = np.asarray(expected['confusion']).ravel().tolist()
            page.wait_for_function(
                'cm => JSON.stringify(prExplorer.state.current.confusion) === JSON.stringify(cm)', arg=flat)
            actual = page.evaluate('prExplorer.state.current')
            for key in ['oa', 'miou', 'mean_recall', 'mean_precision', 'precision', 'recall', 'iou']:
                np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-12)
            assert actual['predicted'] == expected['predicted']
            checked.append(name)

        bias = np.zeros(20)
        verify(bias, 'Unchanged baseline')
        assert page.evaluate('prExplorer.state.testSnapshot') is None
        assert page.locator('#view-test').is_disabled()
        assert page.evaluate('document.getElementById("plot").data[3].x.length') == 20
        sliders = page.locator('#controls input[type="range"]')
        assert sliders.count() == 20
        sliders.nth(1).evaluate("(el) => { el.value = '1.5'; el.dispatchEvent(new Event('input', {bubbles:true})); }")
        bias[1] = 1.5
        verify(bias, 'Durum slider')
        page.locator('#controls .bias').nth(4).fill('-0.75')
        page.locator('#controls .bias').nth(4).dispatch_event('change')
        bias[4] = -.75
        verify(bias, 'Coupled class adjustments')

        def verify_test(biases, name):
            expected = metrics(test_labels, (test_logits + biases).argmax(1))
            page.click('#evaluate-test')
            page.wait_for_function("document.getElementById('metric-split').textContent === 'Test metric'")
            actual = page.evaluate('prExplorer.state.testSnapshot.metrics')
            assert actual['confusion'] == np.asarray(expected['confusion']).ravel().tolist()
            for key in ['oa', 'miou', 'mean_recall', 'precision', 'recall', 'iou']:
                np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-12)
            np.testing.assert_allclose(page.evaluate('document.getElementById("plot").data[3].x'),
                                       expected['recall'], rtol=0, atol=1e-12)
            assert page.locator('#metric-miou').inner_text() == f"{100*expected['miou']:.2f}%"
            assert page.evaluate('prExplorer.state.testMatchesCurrent')
            checked.append(name)

        verify_test(bias, 'Explicit test evaluation matches Python')
        captured = page.evaluate('prExplorer.state.testSnapshot')
        page.evaluate('prExplorer.setBias(1, 1.6)')
        bias[1] = 1.6
        verify(bias, 'Slider edits return to validation')
        page.wait_for_function("document.getElementById('metric-split').textContent === 'Validation metric'")
        assert page.evaluate('prExplorer.state.testSnapshot') == captured
        page.click('#view-test')
        page.wait_for_function("document.getElementById('metric-split').textContent === 'Test metric'")
        assert 'earlier biases' in page.locator('#test-status').inner_text()
        assert page.evaluate('prExplorer.state.testSnapshot') == captured
        stale = page.evaluate('prExplorer.configuration()')
        assert not stale['test_evaluation']['matches_current_biases']
        assert stale['test_evaluation']['biases']['Winter Durum Wheat'] == 1.5
        assert stale['biases']['Winter Durum Wheat'] == 1.6
        page.evaluate('prExplorer.setBias(1, 1.5)')
        bias[1] = 1.5
        verify(bias, 'Return to evaluated biases')
        checked.append('Test snapshot remains fixed until button press')
        with page.expect_download() as download:
            page.click('#export')
        download.value.save_as(a.data / 'browser_export_check.json')
        exported = json.loads((a.data / 'browser_export_check.json').read_text())
        assert exported['split'] == 'val'
        assert exported['biases']['Winter Durum Wheat'] == 1.5
        assert exported['biases']['Winter Cereal'] == -.75
        assert exported['metrics']['confusion'] == page.evaluate('prExplorer.state.current.confusion')
        assert exported['test_evaluation']['split'] == 'test'
        assert exported['test_evaluation']['matches_current_biases']
        assert exported['test_evaluation']['metrics'] == captured['metrics']
        if page.locator('#automatic').is_visible():
            page.click('#automatic')
            automatic = page.evaluate('EXPLORER_DATA.automatic')
            verify(np.array(automatic['biases']), 'Automatic search preset')
            assert page.evaluate('prExplorer.state.biases') == automatic['biases']
            verify_test(np.array(automatic['biases']), 'Automatic configuration on test')
            page.screenshot(path=str(a.data / 'desktop-test.png'), full_page=True)
            page.click('#view-validation')
            page.wait_for_function("document.getElementById('metric-split').textContent === 'Validation metric'")
        page.screenshot(path=str(a.data / 'desktop.png'), full_page=True)
        page.click('#reset')
        bias.fill(0)
        verify(bias, 'Reset restores exact baseline')
        verify_test(bias, 'Zero-bias test baseline')
        page.evaluate('() => { for (let c=0; c<20; c++) prExplorer.setBias(c, 0.7); }')
        bias.fill(.7)
        verify(bias, 'Common bias preserves predictions')
        page.click('#reset')
        page.locator('#controls .bias').nth(1).fill('100')
        page.locator('#controls .bias').nth(1).dispatch_event('change')
        bias.fill(0); bias[1] = 100
        verify(bias, 'Extreme bias and zero-prediction classes')
        verify_test(bias, 'Extreme bias on test')
        assert not page.evaluate('prExplorer.state.current.precision.some(x => !Number.isFinite(x))')
        page.click('#reset')
        page.set_viewport_size({'width': 700, 'height': 1000})
        page.wait_for_timeout(300)
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        page.screenshot(path=str(a.data / 'narrow.png'), full_page=True)
        assert not errors, errors
        assert not requests, requests
        browser.close()
    result = dict(status='passed', checks=checked + ['Configuration download', 'Offline loading',
                 'Responsive layout', 'No browser errors'], metric_absolute_tolerance=1e-12)
    (a.data / 'browser_checks.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
