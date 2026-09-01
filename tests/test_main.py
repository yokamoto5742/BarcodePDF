"""P2: 起動処理の配線"""

from pytest_mock import MockerFixture

import main


def test_main_initializes_logging_before_gui(mocker: MockerFixture) -> None:
    manager = mocker.MagicMock()
    manager.attach_mock(mocker.patch('main.setup_logging'), 'setup_logging')
    manager.attach_mock(mocker.patch('main.tk.Tk'), 'Tk')
    manager.attach_mock(mocker.patch('main.PDFProcessorApp'), 'PDFProcessorApp')

    main.main()

    assert [call[0] for call in manager.mock_calls[:3]] == ['setup_logging', 'Tk', 'PDFProcessorApp']


def test_main_starts_event_loop_with_root_window(mocker: MockerFixture) -> None:
    mocker.patch('main.setup_logging')
    tk_class = mocker.patch('main.tk.Tk')
    app_class = mocker.patch('main.PDFProcessorApp')

    main.main()

    app_class.assert_called_once_with(tk_class.return_value)
    tk_class.return_value.mainloop.assert_called_once()
