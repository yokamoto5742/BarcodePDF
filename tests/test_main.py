"""P2: 起動処理の配線"""

from pytest_mock import MockerFixture

import main


def test_main_initializes_logging_before_gui(mocker: MockerFixture) -> None:
    manager = mocker.MagicMock()
    manager.attach_mock(mocker.patch('main.setup_logging'), 'setup_logging')
    manager.attach_mock(mocker.patch('main.AppConfig'), 'AppConfig')
    manager.attach_mock(mocker.patch('main.cleanup_error_pdfs'), 'cleanup_error_pdfs')
    manager.attach_mock(mocker.patch('main.tk.Tk'), 'Tk')
    manager.attach_mock(mocker.patch('main.PDFProcessorApp'), 'PDFProcessorApp')

    main.main()

    assert [call[0] for call in manager.mock_calls[:5]] == [
        'setup_logging', 'AppConfig', 'cleanup_error_pdfs', 'Tk', 'PDFProcessorApp'
    ]


def test_main_cleans_up_error_pdfs_explicitly(mocker: MockerFixture) -> None:
    """エラーPDFの削除はログ設定の副作用ではなく、起動処理から明示的に呼ぶ"""
    mocker.patch('main.setup_logging')
    mocker.patch('main.tk.Tk')
    mocker.patch('main.PDFProcessorApp')
    config_class = mocker.patch('main.AppConfig')
    cleanup = mocker.patch('main.cleanup_error_pdfs')

    main.main()

    cleanup.assert_called_once_with(config_class.return_value)


def test_main_starts_event_loop_with_root_window(mocker: MockerFixture) -> None:
    mocker.patch('main.setup_logging')
    mocker.patch('main.AppConfig')
    mocker.patch('main.cleanup_error_pdfs')
    tk_class = mocker.patch('main.tk.Tk')
    app_class = mocker.patch('main.PDFProcessorApp')

    main.main()

    app_class.assert_called_once_with(tk_class.return_value)
    tk_class.return_value.mainloop.assert_called_once()
