module Main 
  ( main
  ) where


import           Control.Monad (forM_)
import           Data.Monoid ((<>))
import           Options.Applicative
-- import           Data.Maybe (mapMaybe)
import           Data.Ord (comparing)
import           Data.String (fromString)
import           Data.List (sortBy)
import qualified Data.Map.Strict as M
import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified Data.Text.Lazy.IO as TL

-- import           System.FilePath (isAbsolute, (</>))

-- import           System.Environment (getArgs)

-- import qualified SMT as SMT
-- import qualified Net.Ex1 as Ex1
-- import qualified Net.Ex2 as Ex2
-- import qualified Net.Ex3 as Ex3
-- import qualified Net.Ex4 as Ex4
-- import qualified Net.DAG as DAG
import qualified Net.ArcGraph as Graph
import qualified Net.POS as POS
import qualified Net.MWE2 as MWE
import qualified GradientDescent.Momentum as Mom
import qualified Embedding.Dict as D
import qualified Embedding as Emb


-- main :: IO ()
-- main = do
--   path : depth : _ <- getArgs
--   net <- MWE.train path (read depth) =<< Graph.new 300 2
--   return ()


--------------------------------------------------
-- Commands
--------------------------------------------------


data Command
    = FastText Emb.Config
      -- ^ Calculate fasttext embeddings for the individual words in the file


--------------------------------------------------
-- Parse options
--------------------------------------------------


fastTextOptions :: Parser Command
fastTextOptions = fmap FastText $ Emb.Config
  <$> strOption
        ( metavar "FILE"
       <> long "embed"
       <> short 'e'
       <> help "Fasttext embedding file"
        )
  <*> strOption
        ( metavar "FILE"
       <> long "input"
       <> short 'i'
       <> help "Input .cupt file"
        )
  <*> strOption
        ( metavar "FILE"
       <> long "output"
       <> short 'o'
       <> help "Output file"
        )
  <*> switch
        ( long "no-header"
       <> short 'n'
       <> help "Embedding file has no header"
        )



--------------------------------------------------
-- Global options
--------------------------------------------------


opts :: Parser Command
opts = subparser
  ( command "fasttext"
    (info (helper <*> fastTextOptions)
      (progDesc "Determine fasttext embeddings")
    )
  )


--------------------------------------------------
-- Main
--------------------------------------------------


-- | Run program depending on the cmdline arguments.
run :: Command -> IO ()
run cmd =
  case cmd of
    FastText cfg -> do
      Emb.produceEmbeddings cfg


main :: IO ()
main =
    execParser optsExt >>= run
  where
    optsExt = info (helper <*> opts)
       ( fullDesc
      <> progDesc "MWE identification based on graph-structured neural nets"
      <> header "galago" )
